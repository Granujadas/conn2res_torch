# -*- coding: utf-8 -*-
"""Comprehensive tests for conn2res.connectivity module."""

import torch
import pytest
import networkx as nx
import tempfile
import os
import numpy as np

from conn2res.connectivity import Conn, load_file, get_modules

# Fixtures ----------------------------------------------------------------
"""
Reusable fixtures for device selection and random connectivity generation.
"""

@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return request.param

@pytest.fixture
def random_connectivity(device):
    """Generates random symmetric and asymmetric connectivity matrices."""
    n = 50
    S = torch.rand(n, n, device=device)
    S = (S + S.T) / 2
    S.fill_diagonal_(0)
    for i in range(n):
        S[i, (i+1)%n] = S[(i+1)%n, i] = 1.0  # Ensure ring connectivity
    
    # Asymmetric matrix
    U = torch.rand(n, n, device=device)
    U.fill_diagonal_(0)
    
    return U, S

@pytest.fixture
def conn_objects(random_connectivity, device):
    """Creates Conn instances from random connectivity matrices."""
    U, S = random_connectivity
    return [
        Conn(w=U, device=device),  # Asymmetric
        Conn(w=S, device=device)   # Symmetric
    ]

# Conn Class Tests --------------------------------------------------------
"""
Tests related to the Conn class, including initialization, 
methods for scaling, binarization, randomization, and node selection.
"""

class TestConn:
    @pytest.mark.parametrize("density,expected_tolerance", [
        (None, 0.0), (0.3, 0.15), (0.7, 0.15), (0.9, 0.9)
    ])
    def test_init_density(self, density, expected_tolerance, device):
        """Validates Conn density initialization under different configurations."""
        n = 30
        S = torch.rand(n, n, device=device)
        S = (S + S.T) / 2  # Symmetric
        S.fill_diagonal_(0)
        
        # Add guaranteed connections
        for i in range(n):
            S[i, (i+1)%n] = S[(i+1)%n, i] = 1.0
        
        conn = Conn(w=S, density=density, device=device)
        
        if density is None:
            assert np.isclose(conn.density, (S != 0).sum().item() / (n*(n-1)), 
                            rtol=0.1)
        else:
            assert abs(conn.density - density) <= expected_tolerance

    def test_init_validation(self, device):
        """Checks input validation during Conn initialization."""
        w = torch.rand(5,5, device=device)
        w[1,1] = torch.nan
        w[2,2] = torch.inf
        conn = Conn(w=w, device=device)
        
        assert torch.all(conn.w.diag() == 0)
        assert not torch.any(torch.isnan(conn.w))
        assert not torch.any(torch.isinf(conn.w))
        
        # Test non-square matrix
        with pytest.raises(ValueError):
            Conn(w=torch.rand(3,4, device=device))
            
        # Test invalid type
        with pytest.raises(TypeError):
            Conn(w=np.random.rand(5,5))

    @pytest.mark.parametrize("method", ["scale", "normalize", "scale_and_normalize"])
    def test_scaling_normalization(self, conn_objects, method):
        """Tests scaling and normalization transformations."""
        for conn in conn_objects:
            #original = conn.w.clone()
            getattr(conn, method)()
            
            # Common validations
            assert torch.all(conn.w >= 0), "Negative values after scaling"
            assert torch.all(conn.w <= 1 + 1e-6), "Values exceed 1"
            
            # Spectral radius check
            if "normalize" in method:
                if conn.symmetric:
                    ew = torch.linalg.eigvalsh(conn.w)
                else:
                    ew = torch.linalg.eigvals(conn.w)
                assert torch.allclose(
                    torch.abs(ew).max(), 
                    torch.tensor(1.0, device=conn.device), 
                    atol=1e-3
                )

    def test_binarize(self, conn_objects):
        """Validates that binarization preserves edge count and binary structure."""
        for conn in conn_objects:
            original_edges = conn.n_edges
            conn.binarize()
            
            assert torch.all((conn.w == 0) | (conn.w == 1))
            assert conn.n_edges == original_edges

    @pytest.mark.parametrize("swaps,seed", [(10, 42), (20, 24)])
    def test_randomize(self, swaps, seed, conn_objects):
        """Checks if randomization preserves degree distribution."""
        for conn in conn_objects:
            original_degrees = (conn.w != 0).sum(1)
            conn.randomize(swaps=swaps, seed=seed)
            
            # Validate degree sequence preservation
            randomized_degrees = (conn.w != 0).sum(1)
            assert torch.allclose(
                torch.sort(original_degrees)[0],
                torch.sort(randomized_degrees)[0]
            )

    @pytest.mark.parametrize("threshold", [1, 3])
    def test_threshold(self, threshold, device):
        """Ensures thresholding maintains connectivity and approximates edge count."""
        n = 10
        w = torch.rand(n, n, device=device)
        w = (w + w.T) / 2  # Symmetric
        w.fill_diagonal_(0)
        
        # Add MST connections
        for i in range(n-1):
            w[i, i+1] = w[i+1, i] = 1.0
        
        conn = Conn(w=w, device=device)
        #original_edges = conn.n_edges
        conn.threshold(threshold=threshold)
        
        # Validate connectivity
        G = nx.from_numpy_array(conn.w.cpu().numpy())
        assert nx.is_connected(G)
        
        # Validate edge count
        mst_edges = n - 1
        expected_edges = mst_edges + threshold * mst_edges
        assert conn.n_edges >= expected_edges * 0.9  # 10% tolerance

    @pytest.mark.parametrize("mask,order", [
        ("triu", "random"),
        ("triu", "absrank"),
        ("triu", "rank"),
        ("full", "random"),
        ("full", "absrank"),
        ("full", "rank")
    ])
    def test_add_weights(self, mask, order, device):
        """Tests adding custom weights to connectivity matrices."""
        n = 10
        w_bin = torch.bernoulli(torch.full((n,n), 0.5, device=device))
        w_bin.fill_diagonal_(0)
        conn = Conn(w=w_bin, device=device)
        
        if mask == "triu":
            triu_idx = torch.triu_indices(n, n, 1, device=device)
            n_weights = triu_idx.shape[1]
        else:
            n_weights = conn.n_edges
            
        weights = torch.linspace(0.1, 1.0, n_weights, device=device)
        
        conn.add_weights(weights, mask=mask, order=order, seed=42)
        
        # Validate ordering
        if mask == "triu":
            # Extract only upper triangle values
            applied_weights = conn.w[triu_idx[0], triu_idx[1]]
            
            if order == "absrank":
                sorted_weights = torch.sort(weights.abs(), descending=True)[0]
                assert torch.allclose(
                    applied_weights.abs().sort(descending=True)[0],
                    sorted_weights
                )
            elif order == "rank":
                sorted_weights = torch.sort(weights, descending=True)[0]
                assert torch.allclose(
                    applied_weights.sort(descending=True)[0],
                    sorted_weights
                )
        else:
            # For full mask, validate all weights
            if order == "absrank":
                sorted_weights = torch.sort(weights.abs(), descending=True)[0]
                assert torch.allclose(
                    conn.w[conn.w != 0].abs().sort(descending=True)[0],
                    sorted_weights
                )
            elif order == "rank":
                sorted_weights = torch.sort(weights, descending=True)[0]
                assert torch.allclose(
                    conn.w[conn.w != 0].sort(descending=True)[0],
                    sorted_weights
                )

    def test_subset_nodes(self, conn_objects):
        """Verifies that node subsetting maintains connectivity."""
        for conn in conn_objects:
            original_nodes = conn.n_nodes
            conn.subset_nodes()
            
            # Validate connectivity
            if conn.symmetric:
                G = nx.from_numpy_array(conn.w.cpu().numpy())
                assert nx.is_connected(G)
            else:
                G = nx.DiGraph(conn.w.cpu().numpy())
                assert nx.is_weakly_connected(G)
            
            assert conn.n_nodes <= original_nodes

    @pytest.mark.parametrize("node_set,kwargs", [
        ("shortest_path", {"shortest_path": 2})
    ])
    def test_get_nodes(self, node_set, kwargs, device):
        """Tests node selection methods based on graph distance properties."""
        n = 5 
        w = torch.zeros(n, n, device=device)
        for i in range(n):
            w[i, (i+2) % n] = 1.0
        conn = Conn(w=w, device=device)
        
        nodes = conn.get_nodes(node_set=node_set, **kwargs)
        
        assert nodes.numel() >= 2

# Helper Functions Tests --------------------------------------------------
def test_load_file(device):
    test_tensor = torch.rand(5, 5)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:  # <-- Hardcode .pt
        torch.save(test_tensor, f.name)
        loaded = load_file(f.name, device=device)
    os.remove(f.name)
    
    assert torch.equal(test_tensor.to(device), loaded)
    assert loaded.device == torch.device(device)

def test_get_modules(device):
    module_assignment = torch.tensor([0, 0, 1, 1, 2, 2, 2], device=device)
    module_ids, modules = get_modules(module_assignment)
    
    assert torch.equal(module_ids, torch.tensor([0, 1, 2], device=device))
    assert all(t.device == torch.device(device) for t in modules)
    assert torch.equal(modules[0], torch.tensor([0, 1], device=device))

# Edge Cases and Error Handling -------------------------------------------
@pytest.mark.parametrize("invalid_input", [
    torch.tensor([1,2,3]),        # 1D
    torch.rand(3,3,3),            # 3D
    torch.rand(2,3),              # Non-square
    "invalid_data",               # String
    123                           # Integer
])
def test_invalid_connectivity(invalid_input, device):
    with pytest.raises((ValueError, TypeError)):
        if isinstance(invalid_input, torch.Tensor):
            invalid_input = invalid_input.to(device)
        Conn(w=invalid_input, device=device)

def test_threshold_errors(device):
    # Asymmetric matrix
    w = torch.rand(5,5, device=device)
    conn = Conn(w=w, device=device)
    with pytest.raises(NotImplementedError):
        conn.threshold()

def test_add_weights_errors(conn_objects):
    conn_asym, _ = conn_objects
    invalid_weights = torch.rand(conn_asym.n_edges + 10, device=conn_asym.device)
    
    with pytest.raises(ValueError):
        conn_asym.add_weights(invalid_weights, mask='triu')

# Multi-device Testing ----------------------------------------------------
@pytest.mark.parametrize("method", ["scale", "binarize", "randomize"])
def test_device_consistency(method, device):
    w = torch.rand(10,10, device=device)
    conn = Conn(w=w, device=device)
    getattr(conn, method)()
    assert conn.w.device == torch.device(device)

# Special Matrix Types ----------------------------------------------------
def test_empty_matrix(device):
    w = torch.zeros(0,0, device=device)
    conn = Conn(w=w, device=device)
    assert conn.n_nodes == 0
    assert conn.n_edges == 0
    assert conn.density == 0.0

def test_full_connectivity(device):
    n = 10
    w = torch.ones(n, n, device=device)
    w.fill_diagonal_(0)
    conn = Conn(w=w, device=device)
    assert conn.n_edges == n*(n-1)
    assert np.isclose(conn.density, 1.0, atol=1e-6)