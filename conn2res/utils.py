# -*- coding: utf-8 -*-
"""Provides functions for basic matrix manipulations using PyTorch."""

import torch

# Functions for connectivity matrix
def check_symmetric(a, tol=1e-16):
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    a : torch.Tensor
        Input matrix.
    tol : float, optional
        Tolerance for symmetry check, by default 1e-16.

    Returns
    -------
    bool
        True if the matrix is symmetric, False otherwise.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return False
    
    return torch.allclose(a, a.T, atol=tol)


def make_symmetric(a, copy_lower=True):
    """
    Make a matrix symmetric.

    Parameters
    ----------
    a : torch.Tensor
        Input matrix.
    copy_lower : bool, optional
        If True, use the lower triangle; otherwise, use the upper triangle. Default is True.

    Returns
    -------
    torch.Tensor
        Symmetric matrix.
    """
    if copy_lower:
        return torch.tril(a) + torch.tril(a, diagonal=-1).T
    else:
        return torch.triu(a) + torch.triu(a, diagonal=1).T


def check_square(a):
    """
    Check if a matrix is square.

    Parameters
    ----------
    a : torch.Tensor
        Input matrix.

    Returns
    -------
    bool
        True if the matrix is square, False otherwise.
    """
    return a.ndim == 2 and a.shape[0] == a.shape[1]


# Functions for X and y
def split(a, sections):
    """
    Split a tensor into multiple sections.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor.
    sections : list[int]
        List of indices where to split.

    Returns
    -------
    list of torch.Tensor
        List of split tensors.

    Raises
    ------
    TypeError
        If input is not a tensor.
    """
    if isinstance(a, torch.Tensor):
        return torch.tensor_split(a, tuple(sections), 0)
    else:
        raise TypeError("Input must be a torch.Tensor")


def get_sections(a):
    """
    Get split sections from a list or tuple of tensors.

    Parameters
    ----------
    a : list or tuple
        List of tensors.

    Returns
    -------
    torch.Tensor
        Indices for splitting.

    Raises
    ------
    TypeError
        If input is not a list or tuple.
    """
    if isinstance(a, (list, tuple)):
        lens = torch.tensor([len(i) for i in a])
        sections = torch.cumsum(lens[:-1], dim=0)
        return sections
    else:
        raise TypeError("Input must be a list or tuple of tensors")


def concat(a):
    """
    Concatenate a list or tuple of tensors.

    Parameters
    ----------
    a : list or tuple
        List of tensors.

    Returns
    -------
    torch.Tensor
        Concatenated tensor.

    Raises
    ------
    TypeError
        If input is not a list, tuple, or tensor.
    """
    if isinstance(a, torch.Tensor):
        return a
    elif isinstance(a, (list, tuple)):
        if len(a) == 0:
            raise ValueError("An empty tuple/list cannot be concatenated")
        return torch.cat(a, dim=0)
    else:
        raise TypeError("The input must be a tuple or tensor")


def check(*objs):
    """
    Check if objects share a common base class.

    Parameters
    ----------
    *objs : objects
        Objects to compare.

    Returns
    -------
    bool
        True if objects share a common base class, False otherwise.
    """
    if not objs:
        return True  # Handle empty input case

    def _all_bases(o):
        for b in o.__bases__:
            if b is not object:
                yield b
            yield from _all_bases(b)
    
    s = [(i.__class__, *_all_bases(i.__class__)) for i in objs]
    return len(set(s[0]).intersection(*s[1:])) > 0