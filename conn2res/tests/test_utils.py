# -*- coding: utf-8 -*-
"""Tests for conn2res.utils functions."""

import torch 
import pytest

from conn2res.utils import (
    check_symmetric, 
    make_symmetric, 
    check_square, 
    split, 
    get_sections, 
    concat, 
    check
)

# ========================
# check_symmetric Testing
# ========================
@pytest.mark.parametrize("matrix, tol, expected, note", [
    # Squared matrices
    ([[1.0, 2.0], [2.0, 1.0]], 1e-8, True, "Exact symmetric"),
    ([[1.0, 2.0], [2.0000001, 1.0]], 1e-5, True, "Symmetric within tolerance"),
    ([[1.0, 2.0], [3.0, 4.0]], 1e-8, False, "Non-symmetric"),
    
    # Non-squared matrices
    ([[1, 2, 3], [4, 5, 6]], 1e-8, False, "2x3 non-squared"),
    (torch.randn(5, 3), 1e-8, False, "5x3 random"),
    
    # Special scenarios
    (torch.eye(3), 1e-8, True, "Identity matrix"),
    (torch.zeros(4, 4), 1e-8, True, "All zeros"),
    
    # High dimentionality tensors
    (torch.randn(2, 2, 2), 1e-8, False, "3D tensor"),
])
def test_check_symmetric(matrix, tol, expected, note):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
        
    assert check_symmetric(matrix, tol) == expected

# ========================
# make_symmetric Testing
# ========================
@pytest.mark.parametrize("matrix, copy_lower, expected_note", [
    # Base cases
    ([[1, 2], [3, 4]], True, "lower triangular copy"),
    ([[1, 2], [3, 4]], False, "upper triangular copy"),
    
    # Already symmetric
    ([[5, 5], [5, 5]], True, "already symmetric"),
    (torch.randn(5, 5).tril() + torch.randn(5, 5).tril().T, False, "force upper"),
    
    # Dimensional stress
    (torch.randn(10, 10), True, "large matrix"),
    (torch.randn(1, 1), False, "1x1 matrix"),
])
def test_make_symmetric(matrix, copy_lower, expected_note):
    a = matrix if isinstance(matrix, torch.Tensor) else torch.tensor(matrix, dtype=torch.float32)
    
    result = make_symmetric(a, copy_lower)
    
    # Fundamental property: matrix must be symmetric
    assert torch.allclose(result, result.T, atol=1e-8), "Resulting matrix must be symmetric"
    
    # Validation of chosen triangle and transpose matrix
    if copy_lower:
        # Check that the lower triangle matches the original
        assert torch.allclose(torch.tril(result), torch.tril(a)), "The lower triangle does not match the original"
        # Check that the upper triangle is the transpose of the lower (excluding diagonal)
        assert torch.allclose(torch.triu(result, diagonal=1), torch.tril(a, diagonal=-1).T), "The upper triangle is not the correct transpose"
    else:
        # Check that the upper triangle matches the original
        assert torch.allclose(torch.triu(result), torch.triu(a)), "The upper triangle does not match the original"
        # Check that the lower triangle is the transpose of the upper (excluding diagonal
        assert torch.allclose(torch.tril(result, diagonal=-1), torch.triu(a, diagonal=1).T), "The lower triangle is not the correct transpose"

# ========================
# check_square Testing
# ========================
@pytest.mark.parametrize("shape, expected", [
    ((5, 5), True),
    ((3, 4), False),
    ((1, 1), True),
    ((0, 0), True),  # Edge case
    ((2, 3, 2), False),  # 3D tensor
])
def test_check_square(shape, expected):
    tensor = torch.empty(shape) if 0 in shape else torch.randn(*shape)
    assert check_square(tensor) == expected

# ========================
# split Testing
# ========================
@pytest.mark.parametrize("tensor, sections, expected_shapes", [
    # 1D cases
    (torch.arange(5), [3], [(3,), (2,)]),
    (torch.tensor([1]), [], [(1,)]),  # No split
    
    # 2D+ cases
    (torch.arange(12).reshape(4, 3), [2], [(2, 3), (2, 3)]),
    (torch.randn(6, 2, 2), [3, 5], [(3, 2, 2), (2, 2, 2), (1, 2, 2)]),
    
    # Special cases
    (torch.empty(0), [], [(0,)]),  # Empty tensor
])
def test_split(tensor, sections, expected_shapes):
    result = split(tensor, sections)
    assert len(result) == len(expected_shapes)
    for t, shape in zip(result, expected_shapes):
        assert t.shape == torch.Size(shape)

@pytest.mark.parametrize("invalid_input", [
    [1, 2, 3],  # Python list
    "not a tensor",  # String
    None,  # NoneType
])
def test_split_errors(invalid_input):
    with pytest.raises(TypeError):
        split(invalid_input, [1])

# ========================
# get_sections Testing
# ========================
@pytest.mark.parametrize("tensors, expected_sections", [
    ([torch.tensor([1, 2])], []),
    ([torch.tensor([1]), torch.tensor([])], [1]),  
    ([torch.arange(3), torch.ones(2), torch.zeros(5)], [3, 5]),
    # Multidimensional tensors
    ([torch.randn(2, 3), torch.randn(4, 3)], [2]),
])
def test_get_sections(tensors, expected_sections):
    sections = get_sections(tensors)
    assert torch.equal(sections, torch.tensor(expected_sections))

@pytest.mark.parametrize("invalid_input", [
    torch.tensor([1]),  # Direct tensor
    123,  # Integer
    {"key": "value"},  # Dictionary
])
def test_get_sections_errors(invalid_input):
    with pytest.raises(TypeError):
        get_sections(invalid_input)

# ========================
# concat Testing
# ========================
@pytest.mark.parametrize("inputs, expected_shape", [
    # Standard cases
    ([torch.tensor([[1, 2]])], (1, 2)),
    ([torch.ones(3, 2), torch.zeros(2, 2)], (5, 2)),

    # Input types
    ((torch.tensor([1]), torch.tensor([2])), (2,)),  # Tuple
    (torch.tensor([[1, 2]]), (1, 2)),  # Single tensor
    
    # Edge cases
    ([], "ValueError"),  # Empty list
])
def test_concat(inputs, expected_shape):
    if expected_shape == "ValueError":
        with pytest.raises(ValueError):
            concat(inputs)
    else:
        result = concat(inputs)
        assert result.shape == torch.Size(expected_shape)

@pytest.mark.parametrize("invalid_input", [
    123,
    ["not a tensor", 456],
    (torch.tensor([1]), "string"),
    [],
])
def test_concat_errors(invalid_input):
    if isinstance(invalid_input, (list, tuple)) and len(invalid_input) == 0:
        with pytest.raises(ValueError):
            concat(invalid_input)
    else:
        with pytest.raises(TypeError):
            concat(invalid_input)

# ========================
# check Testing
# ========================
class BaseTest: 
    pass
class ATest(BaseTest): 
    pass
class BTest(BaseTest): 
    pass
class CTest: 
    pass

@pytest.mark.parametrize("objects, expected", [
    # Inheritance cases
    ([ATest(), BTest()], True),  # Share BaseTest
    ([ATest(), ATest()], True),  # Same class
    
    # Non common inheritance
    ([ATest(), CTest()], False),
    ([torch.Tensor([1]), 123], False),  # Tensor vs int
    
    # Special cases
    ([], True),  # Empty list
    ([ATest()], True),  # Only one element
])
def test_check(objects, expected):
    if len(objects) == 0:
        assert check(*objects) == expected  # empty check() is technically True
    else:
        assert check(*objects) == expected