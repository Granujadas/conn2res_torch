# -*- coding: utf-8 -*-
""" Functionality for connectivity matrix. """
import os

import torch
import numpy as np

from conn2res import utils


PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_DIR, 'examples', 'data', 'human')


class Conn():
    """
    Class that represents a weighted or unweighted network using connectivity data

    Notes:
    - Diagonal is set to zero and non-finite values are removed
    - Handles both symmetric and asymmetric connectivity matrices
    - Implements graph theory operations and connectivity manipulations
    """

    def __init__(self, w=None, filename=None, subj_id=0, modules=None, density=None, device='cpu'):
        """
        Parameters
        ----------
        w : torch.Tensor, optional
            Connectivity matrix (source, target).
        filename : str, optional
            Filename of the connectivity matrix to be loaded from disk.
            specified using full path (by default, the connectivity matrix is
            loaded from examples/data)
        subj_id : int, optional
            Index of subject in group-level connectivity data.
        modules : torch.Tensor, optional
            Array storing module assignments for each node.
        density : float, optional
            Density to which the network should be set. (note that
            connectedness in not checked during this process, so it should
            be used with care!)
        device : str, optional
            Device to place tensors on ('cpu' or 'cuda').
        """
        if w is not None and not isinstance(w, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if w is not None and (w.ndim != 2 or w.shape[0] != w.shape[1]):
            raise ValueError("Connectivity matrix must be 2D and square.")

        self.device = device

        # Load or initialize the connectivity matrix
        if w is not None:
            # assign provided connectivity data
            self.w = torch.as_tensor(w, dtype=torch.float32, device=device)
        else:
            # load connectivity data
            if filename is not None:
                self.w = torch.load(filename, map_location=device)
            else:
                self.w = self._load_file('connectivity.npy', map_location=device)

            # select one subject from group connectivity data
            if subj_id is not None and self.w.ndim == 3:
                self.w = self.w[:, :, subj_id]

        # Set zero diagonal & remove inf/nan values
        self.w.fill_diagonal_(0)
        self.w[torch.logical_or(torch.isinf(self.w), torch.isnan(self.w))] = 0

        # Initialize attributes
        self.n_nodes = self.w.shape[0]
        self.n_edges = (self.w != 0).sum().item()
        self.symmetric = utils.check_symmetric(self.w)
        self.idx_node = torch.ones(self.n_nodes, dtype=torch.bool, device=device)
        self.modules = modules.to(device) if modules is not None else None

        # Handle density
        if self.n_nodes > 1:
            self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1))
        else:
            self.density = 0.0

        if density is not None:
            self._set_density(density)
            self.subset_nodes()

    def scale_and_normalize(self):
        """ Scale the connectivity matrix to [0, 1] and normalize by spectral radius """
        self.scale()
        self.normalize()

    def scale(self):
        """ Scale weights to [0, 1] """
        if self.w.numel() == 0:
            return
            
        w_min = self.w.min()
        w_max = self.w.max()
        if w_max - w_min > 1e-8:
            self.w = (self.w - w_min) / (w_max - w_min)
        else:
            self.w = torch.zeros_like(self.w)

    def normalize(self):
        """ Normalize the connectivity matrix by spectral radius """
        if self.w.numel() == 0:
            return
            
        if self.symmetric:
            ew = torch.linalg.eigvalsh(self.w)
        else:
            ew = torch.linalg.eigvals(self.w)
            
        spectral_radius = torch.abs(ew).max()
        if spectral_radius > 1e-8:
            self.w /= spectral_radius

    def binarize(self):
        """ Binarize the connectivity matrix """
        self.w = (self.w != 0).float()

    def randomize(self, swaps=10, seed=None):
        """
        Randomize the connectivity matrix

        Parameters
        ----------
        swaps : int, optional
            Indicates the amount of swaps at the randomizing process.
        seed : int, optional
            Enables experimental replicability.
        """

        if seed is not None:
            torch.manual_seed(seed)

        if self.symmetric:
            self._randomize_undirected(swaps)
        else:
            self._randomize_directed(swaps)

    def threshold(self, threshold=1):
        """
        Threshold the connectivity matrix by keeping the network connected.
        Works only for undirected networks.

        Parameters
        ----------
        threshold : int, optional
            Establishes threshold multiplier for minimum spanning tree edges.
        """

        if not self.symmetric:
            raise NotImplementedError("Thresholding only implemented for symmetric matrices")
        
        # Create binary matrix
        w_bin = (self.w != 0).float()
        
        # Get minimum spanning tree (MST)
        mst = self._get_minimum_spanning_tree(w_bin)
        
        # Calculate number of edges to keep
        mst_edges = (mst != 0).sum().item() // 2
        n_keep = mst_edges + threshold * mst_edges
        n_keep = min(n_keep, self.n_edges)

        if n_keep > 0:
            triu_weights = self.w.triu(1)
            edge_weights = triu_weights[triu_weights != 0]
            
            if len(edge_weights) > 0:
                threshold_val = torch.kthvalue(edge_weights, len(edge_weights) - n_keep + 1).values
                self.w[(self.w < threshold_val) & (mst == 0)] = 0
                self.w = utils.make_symmetric(self.w, copy_lower=False)

        self.n_edges = (self.w != 0).sum().item()
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1)) if self.n_nodes > 1 else 0.0

    def add_weights(self, weights, mask='triu', order='random', seed=None):
        """
        Add weights to either a binary or weighted connectivity matrix

        Parameters
        ----------
        weights : torch.tensor
            the weights to be added to the connectivity matrix
        mask : str, optional
            mask to be used to replace weights in the connectivity matrix, by
            default triu (upper triangular matrix)
        order : str, optional
            it decides whether the weights should be added randomly to the
            connectivity matrix or for instance, the rank of the weights
            should be kept, by default random
        seed : int, array_like[ints], SeedSequence, BitGenerator, Generator, optional
            seed to initialize the random number generator, by default None
            for details, see numpy.random.default_rng()
        """

        if mask not in ['triu', 'full']:
            raise ValueError("Mask must be 'triu' or 'full.'")

        # use random number generator for reproducibility
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)

        if mask == 'full':
            if weights.numel() != self.n_edges:
                raise ValueError(f"Expected {self.n_edges} weights for full mask")
                
            edge_indices = torch.nonzero(self.w)
            if order == 'random':
                perm = torch.randperm(self.n_edges, generator=rng)
                weights = weights[perm]
            elif order == 'absrank':
                weights = weights[torch.argsort(torch.abs(weights), descending=True)]
            elif order == 'rank':
                weights = weights[torch.argsort(weights, descending=True)]
                
            self.w[edge_indices[:,0], edge_indices[:,1]] = weights

        elif mask == 'triu':
            triu_idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
            n_edges = triu_idx.shape[1]
            if weights.numel() != n_edges:
                raise ValueError(f"Expected {n_edges} weights for triu mask")
                
            if order == 'random':
                weights = weights[torch.randperm(n_edges, generator=rng)]
            elif order == 'absrank':
                weights = weights[torch.argsort(torch.abs(weights), descending=True)]
            elif order == 'rank':
                weights = weights[torch.argsort(weights, descending=True)]
                
            self.w[triu_idx[0], triu_idx[1]] = weights
            self.w = utils.make_symmetric(self.w, copy_lower=False)

    def subset_nodes(self, node_set='all', idx_node=None):
        """
        Reduce the connectivity matrix to a subset of nodes

        By default, the connectivity matrix is reduced to the largest
        connected component

        Parameters
        ----------
        node_set : str, optional
            subset of nodes defined as string, by default all
        idx_node : torch.Tensor (dtype=bool), optional
            boolean indexes of nodes to be used for subset of nodes
        """

        if idx_node is None:
            idx_node = self._get_largest_component()
            
        if idx_node.dtype != torch.bool or idx_node.shape != (self.n_nodes,):
            raise ValueError("Invalid node indices")
            
        # Update attributes
        self.idx_node = idx_node
        self.w = self.w[idx_node, :][:, idx_node]
        self.n_nodes = idx_node.sum().item()
        self.n_edges = (self.w != 0).sum().item()
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1)) if self.n_nodes > 1 else 0.0
        
        if self.modules is not None:
            self.modules = self.modules[idx_node]

    def get_nodes(self, node_set='all', nodes_from=None, nodes_without=None, **kwargs):
        """
        Get a set of nodes from the connectivity matrix

        Parameters
        ----------
        node_set : str
            Predefined subset type ('all', 'ctx', 'subctx', 'random', 'shortest_path', or module name).
        nodes_from : torch.Tensor, optional
            Nodes to consider for selection.
        nodes_without : torch.Tensor, optional
            Nodes to exclude from selection

        Returns
        -------
        torch.Tensor
            Indices of selected nodes.
        """

        if nodes_from is None:
            nodes_from = torch.arange(self.n_nodes, device=self.device)
        else:
            nodes_from = torch.as_tensor(nodes_from, device=self.device)
            
        if nodes_without is not None:
            nodes_without = torch.as_tensor(nodes_without, device=self.device)
            mask = ~torch.isin(nodes_from, nodes_without)
            nodes_from = nodes_from[mask]
            
        if node_set == 'all':
            return nodes_from
            
        elif node_set in ['ctx', 'subctx']:
            if self.modules is None:
                raise ValueError("Module assignments required for cortical selection")
                
            ctx_values = self.modules[nodes_from]
            target_value = 1 if node_set == 'ctx' else 0
            return nodes_from[ctx_values == target_value]
            
        elif node_set == 'random':
            perm = torch.randperm(len(nodes_from), device=self.device)
            return nodes_from[perm[:kwargs.get('n_nodes', 1)]]
            
        elif node_set == 'shortest_path':
            D = self._distance_bin_torch()
            pairs = torch.combinations(nodes_from, 2)
            valid_pairs = pairs[D[pairs[:,0], pairs[:,1]] == kwargs['shortest_path']]
            return torch.unique(pairs[valid_pairs].flatten())
            
        else:
            if self.modules is None:
                raise ValueError("Module assignments required for module selection")
                
            return nodes_from[self.modules[nodes_from] == node_set]

    def _get_largest_component(self):
        """ Find the largest connected component using BFS """

        if self.n_nodes == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
            
        if self.symmetric:
            adj = (self.w + self.w.T) > 0
        else:
            adj = (self.w + self.w.T).bool()

        components = -torch.ones(self.n_nodes, dtype=torch.long, device=self.device)
        current_component = 0
        
        for node in range(self.n_nodes):
            if components[node] == -1:
                queue = [node]
                components[node] = current_component
                
                while queue:
                    current = queue.pop(0)
                    neighbors = torch.nonzero(adj[current], as_tuple=False).flatten()
                    
                    for neighbor in neighbors:
                        if components[neighbor] == -1:
                            components[neighbor] = current_component
                            queue.append(neighbor)
                            
                current_component += 1
                
        component_sizes = torch.bincount(components[components != -1])
        if len(component_sizes) == 0:
            return torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
            
        largest_component = torch.argmax(component_sizes)
        return components == largest_component

    def _update_attributes(self, idx_node):
        """
        Update network attributes after a subset nodes operation.

        Parameters
        ----------
        idx_node : torch.Tensor (dtype=bool)
            boolean indexes of nodes which we want to changes the attributes

        Raises
        ------
        ValueError
            Boolean indexing should be used for nodes
        """

        # Verify the correct format for boolean Tensor
        if not (isinstance(idx_node, torch.Tensor) and idx_node.dtype == torch.bool):
            raise ValueError("Boolean indexing should be used for nodes")

        # Update node attributes
        self.idx_node[self.idx_node.clone()] = idx_node.to(self.device)

        # Update connectivity matrix
        mask = idx_node.unsqueeze(1) & idx_node.unsqueeze(0)
        self.w = self.w[mask].view((idx_node.sum(), -1))

        # Update numeric attributes
        self.n_nodes = int(idx_node.sum().item())
        self.n_edges = torch.sum(self.w != 0).item()
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1 + 1e-8))
        
    def _set_density(self, density):
        """Adjust matrix density while preserving connectedness"""
        if density < 0 or density > 1:
            raise ValueError("Density must be between 0 and 1")

        if self.symmetric:
            triu_vals = self.w.triu(1).flatten()
            non_zero = triu_vals[triu_vals != 0]
            max_possible = len(non_zero)
            n_keep = int((self.n_nodes * (self.n_nodes - 1) // 2) * density)
            n_keep = min(n_keep, max_possible)
            
            if n_keep > 0:
                threshold_idx = max(0, len(non_zero) - n_keep)
                threshold = torch.kthvalue(non_zero, threshold_idx).values
                self.w[(self.w < threshold) & (self.w != 0)] = 0
                self.w = utils.make_symmetric(self.w, copy_lower=False)
        else:
            vals = self.w.flatten()
            non_zero = vals[vals != 0]
            n_keep = int(self.n_nodes * (self.n_nodes - 1) * density)
            n_keep = min(n_keep, len(non_zero))
            
            if n_keep > 0:
                threshold_idx = max(0, len(non_zero) - n_keep)
                threshold = torch.kthvalue(non_zero, threshold_idx).values
                self.w[self.w < threshold] = 0

        self.n_edges = (self.w != 0).sum().item()
        self.density = self.n_edges / (self.n_nodes * (self.n_nodes - 1)) if self.n_nodes > 1 else 0.0

    def _randomize_undirected(self, swaps):
        # Get edges' indexes at triangular upper
        edge_index = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        edge_mask = self.w[edge_index[0], edge_index[1]] != 0
        edges = edge_index[:, edge_mask].T
        
        for _ in range(swaps):
            if len(edges) < 2:
                break
                
            # Select two different edges
            idx = torch.randperm(len(edges))[:2]
            a, b = edges[idx[0]]
            c, d = edges[idx[1]]
            
            # Verify the new edges did not previously exist
            if (a != c) and (b != d) and (self.w[a, c] == 0) and (self.w[b, d] == 0):
                # Swap edges mantaining symmetry 
                self.w[a, b] = self.w[b, a] = 0
                self.w[c, d] = self.w[d, c] = 0
                self.w[a, c] = self.w[c, a] = 1
                self.w[b, d] = self.w[d, b] = 1

                # Update edges
                edges[idx[0]] = torch.tensor([a, c], device=self.device)
                edges[idx[1]] = torch.tensor([b, d], device=self.device)

    def _randomize_directed(self, swaps):
        """Edge randomization for directed networks"""
        edge_sources, edge_targets = torch.nonzero(self.w, as_tuple=True)
        edges = torch.stack([edge_sources, edge_targets], dim=1)
        
        for _ in range(swaps):
            if len(edges) < 2:
                break
                
            # Select two random edges (A→B and C→D)
            idx = torch.randint(0, len(edges), (2,), device=self.device)
            a, b = edges[idx[0]]
            c, d = edges[idx[1]]
            
            # Check if new edges (A→D and C→B) don't exist
            if (a != c) and (b != d) and (self.w[a, d] == 0) and (self.w[c, b] == 0):
                # Perform swap
                self.w[a, b] = 0
                self.w[c, d] = 0
                self.w[a, d] = 1
                self.w[c, b] = 1
                
                # Update edges list
                edges[idx[0]] = torch.tensor([a, d], device=self.device)
                edges[idx[1]] = torch.tensor([c, b], device=self.device)

    def _get_minimum_spanning_tree(self, w_bin):
        """Prim's algorithm for MST"""
        device = self.device
        n_nodes = self.n_nodes
        
        # Initialize
        mst = torch.zeros_like(w_bin)
        visited = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        visited[0] = True
        
        for _ in range(n_nodes - 1):
            # Find edges between visited and unvisited nodes
            border_edges = w_bin * visited.unsqueeze(1) * ~visited.unsqueeze(0)
            src, dst = torch.nonzero(border_edges, as_tuple=True)
            
            if len(src) == 0:
                break
                
            # Add strongest available edge
            max_idx = torch.argmax(self.w[src, dst])
            mst[src[max_idx], dst[max_idx]] = 1
            mst[dst[max_idx], src[max_idx]] = 1
            visited[dst[max_idx]] = True
        
        return mst

    def _distance_bin_torch(self):
        """Calculate binary distance matrix using Floyd-Warshall"""
        if self.n_nodes == 0:
            return torch.empty((0, 0), device=self.device)
        
        # Usar tensores en el dispositivo correcto
        ones = torch.tensor(1.0, device=self.device)
        infs = torch.tensor(float('inf'), device=self.device)
        
        D = torch.where(self.w != 0, ones, infs)
        D.fill_diagonal_(0)
        
        for k in range(self.n_nodes):
            D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
            
        return D

    def _load_file(self, filename, map_location='cpu'):
        """Load data from supported file formats"""
        filepath = os.path.join(DATA_DIR, filename)
        
        if filename.endswith('.npy'):
            data = np.load(filepath)
            return torch.from_numpy(data).to(map_location)
        elif filename.endswith('.pt'):
            return torch.load(filepath, map_location=map_location)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def __repr__(self):
        """This method returns datatype valuable information"""
        return f"Conn(n_nodes={self.n_nodes}, density={self.density:.2f}, symmetric={self.symmetric})"
   

def load_file(filename, device='cpu'):
    """
    Load data from disc

    Parameters
    ----------
    filename : str
        Filename of the data to be loaded from DATA_DIR in disc.
    device: str, optional
        Device in which the tensor will be loaded ('cpu' or 'cuda').

    Returns
    -------
    result : torch.tensor
        Data stored in the file.
    """
    return Conn._load_file(None, filename, device)

def get_modules(module_assignment):
    """
    Get module assignment of nodes

    Parameters
    ----------
    module_assignment : torch.Tensor
        1D Tensor containing the assignation of modules for each node.

    Returns
    -------
    module_ids : torch.Tensor
        1D Tensor of unique modules.
    readout_modules : list[torch.Tensor]
        List if tensors with node indexes for each module.
    """
    module_ids = torch.unique(module_assignment)
    readout_modules = [torch.where(module_assignment == uid)[0] for uid in module_ids]
    return module_ids, readout_modules