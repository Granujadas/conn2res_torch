# -*- coding: utf-8 -*-
"""
For testing conn2res.tasks functionality
"""

import torch
import pytest

from conn2res.tasks import (
    NeuroGymTask, 
    ReservoirPyTask, 
    Conn2ResTask
)

# ====================
# NeuroGymTask Testing
# ====================
@pytest.mark.parametrize("task_name", [
    'AntiReach',
    'ContextDecisionMaking',
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    'DualDelayMatchSample',
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming',
    'MultiSensoryIntegration',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D',
    'Reaching1DWithSelfDistraction',
    'ReachingDelayResponse',
    'ReadySetGo',
    'SingleContextDecisionMaking'
])
def test_neurogym_task(task_name):
    task = NeuroGymTask(name=task_name, n_trials=2)
    x, y = task.fetch_data(input_gain=2.0, add_bias=True)
    
    assert isinstance(x, list) and isinstance(y, list)
    assert all(isinstance(xi, torch.Tensor) for xi in x)
    assert all(isinstance(yi, torch.Tensor) for yi in y)
    assert x[0].shape[0] == y[0].shape[0]
    assert task.n_features == x[0].shape[1]
    assert task.n_targets == y[0].shape[1]

# ========================
# ReservoirPyTask Testing
# ========================
# TODO: Fix bugs for chaotic series (lorenz, multiscroll, doublescroll, rabinovich_fabrikant, lonrenz96, rossler) 
@pytest.mark.parametrize("task_name", [
    'henon_map',
    'logistic_map',
    # 'lorenz',
    'mackey_glass',
    # 'multiscroll',
    # 'doublescroll',
    # 'rabinovich_fabrikant',
    'narma',
    # 'lorenz96',
    # 'rossler'
])
def test_reservoirpy_task(task_name):
    task = ReservoirPyTask(name=task_name, n_trials=100)
    x, y = task.fetch_data(horizon=[1, 2], input_gain=1.0, add_bias=True)
    
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == y.shape[0]
    assert task.n_features == x.shape[1]
    assert task.n_targets == y.shape[1]

# ========================
# Conn2ResTask Testing
# ========================
def test_conn2res_task():
    task = Conn2ResTask(name="MemoryCapacity", n_trials=150)
    x, y = task.fetch_data(horizon_max=-5, input_gain=1.0, add_bias=True, seed=42)

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == y.shape[0]
    assert task.n_features == x.shape[1]
    assert task.n_targets == y.shape[1]