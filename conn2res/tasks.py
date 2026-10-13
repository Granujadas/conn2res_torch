# -*- coding: utf-8 -*-
"""
Functionality for fetching task datasets
"""
from abc import ABCMeta, abstractmethod
import neurogym as ngym
from reservoirpy import datasets
import torch

# Tasks list
NEUROGYM_TASKS = [
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
]

RESERVOIRPY_TASKS = [
    'henon_map',
    'logistic_map',
    'lorenz', # *
    'mackey_glass',
    'multiscroll', # *
    'doublescroll', # *
    'rabinovich_fabrikant', # *
    'narma',
    'lorenz96', # *
    'rossler' # *
]

CONN2RES_TASKS = [
    'MemoryCapacity'
]


class Task(metaclass=ABCMeta):
    """
    Class for generating task datasets

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, name, n_trials=10):
        """
        Constructor method for class Task
        """
        self.name = name
        self.n_trials = n_trials
        self.n_targets = None
        self.n_features = None

    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, name):
        pass

    @abstractmethod
    def fetch_data(self, n_trials=None, **kwargs):
        pass


class NeuroGymTask(Task):
    """
    Class for generating task datasets from the
    Neurogym repository

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor method for class NeuroGymTask
        """
        super().__init__(*args, **kwargs)
        self.timing = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in NEUROGYM_TASKS:
            raise ValueError("Task not included in NeuroGym tasks")

        self._name = name

    def fetch_data(self, n_trials=None, input_gain=None, add_bias=False, **kwargs):
        """
        Fetch data for Neurogym tasks using PyTorch

        Parameters
        ----------
        n_trials : int, optional
            number of trials to be generated, by default None
        input_gain : float, optional
            gain on the input signal, i.e., scalar multiplier, by default None
        add_bias : bool, optional
            decides whether bias is added to the input signal or not,
            by default False

        Returns
        -------
        x, y : list of torch.Tensor
            input (x) and output (y) training data per trial
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # Create environment
        dataset = ngym.Dataset(self._name + '-v0', kwargs)
        env = dataset.env
        _ = env.reset()

        x_list, y_list = [], []
        for _ in range(self.n_trials):
            env.new_trial()
            ob, gt = env.ob, env.gt

            # Convert to PyTorch tensors
            ob = torch.tensor(ob, dtype=torch.float32)
            gt = torch.tensor(gt, dtype=torch.float32)

            # Ensure shape [T, 1]
            if ob.ndim == 1:
                ob = ob.unsqueeze(-1)
            if gt.ndim == 1:
                gt = gt.unsqueeze(-1)

            # Apply input gain
            if input_gain is not None:
                ob = ob * input_gain

            # Add bias term if needed
            if add_bias:
                ones = torch.ones((ob.size(0), 1), dtype=torch.float32)
                ob = torch.cat([ones, ob], dim=1)

            x_list.append(ob)
            y_list.append(gt)

        # Set attributes
        self.n_features = x_list[0].size(1)
        self.n_targets = y_list[0].size(1)
        self.timing = env.timing

        return x_list, y_list


class ReservoirPyTask(Task):
    """
    Class for generating task datasets from the
    ReservoirPy repository

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor method for class ReservoirPyTask
        """
        super().__init__(*args, **kwargs)
        self.horizon = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in RESERVOIRPY_TASKS:
            raise ValueError("Task not included in ReservoirPy tasks")

        self._name = name

    def fetch_data(self, n_trials=None, horizon=1, win=30,
                   input_gain=None, add_bias=False, **kwargs):
        """
        Fetch data for ReservoirPy tasks, using PyTorch tensors

        Parameters
        ----------
        n_trials : int, optional
            number of time steps in input and output, by default None
        horizon : int or list of int
            shift(s) between input and output, positive for prediction,
            negative for memory task
        win : int
            initial window of the input signal to be used for generating the
            delayed output signal, by default 30
        input_gain : float
            scalar multiplier for input
        add_bias : bool
            whether to add bias term to input

        Returns
        -------
        x, y : torch.Tensor
            input (x) and output (y) training data
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # Convert horizon to list if needed
        if isinstance(horizon, (int, torch.Tensor)):
            horizon = [horizon]

        # Check that all horizon values have the same sign
        signs = torch.tensor(horizon).sign().unique()
        if signs.numel() != 1:
            raise ValueError("Horizon should have elements with same sign")

        abs_horizon_max = max(abs(h) for h in horizon)
        if win < abs_horizon_max:
            raise ValueError("Absolute maximum horizon should be within window")

        # Generate input data from ReservoirPy
        env = getattr(datasets, self._name)
        x_np = env(n_timesteps=self.n_trials + win + abs_horizon_max + 1, **kwargs)
        x = torch.tensor(x_np, dtype=torch.float32)

        # Reshape if needed
        if x.ndim == 1:
            x = x.unsqueeze(-1)

        # Generate output data with delays
        y_slices = []
        for h in horizon:
            start = win + h
            end = -abs_horizon_max + h - 1 or None
            y_slices.append(x[start:end])
        y = torch.cat(y_slices, dim=1)

        # Trim input data
        x = x[win:-abs_horizon_max - 1]

        # Apply input gain
        if input_gain is not None:
            x = x * input_gain

        # Add bias term
        if add_bias:
            bias = torch.ones((x.size(0), 1), dtype=torch.float32)
            x = torch.cat([bias, x], dim=1)

        # Set attributes
        self.n_features = x.size(1)
        self.n_targets = y.size(1)
        self.horizon = horizon

        return x, y


class Conn2ResTask(Task):
    """
    Class for generating datasets for MemoryCapacity task

    Parameters
    ----------
    name : str
        name of the task
    n_trials : int, optional
        number of trials if task indicated by 'name' is a
        a trial-based task, by default 10
    """

    def __init__(self, *args, **kwargs):
        """
            Constructor method for class Conn2ResTask
        """
        super().__init__(*args, **kwargs)
        self.horizon_max = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name not in CONN2RES_TASKS:
            raise ValueError("Task not included in conn2res tasks")

        self._name = name

    def fetch_data(self, n_trials=None, horizon_max=-20, win=30,
                   low=-1, high=1, input_gain=None, add_bias=False,
                   seed=None):
        """
        Fetch data for MemoryCapacity, which is defined as a multi-output
        task using a uniformly distributed input signal and multiple
        delayed output signals

        Parameters
        ----------
        n_trials : int, optional
            number of time steps in input and output, by default None
        horizon_max : int, optional
            maximum shift between input and output, i.e., negative number 
            for memory capacity task, by default -20
            note that an array of horizons are generated from -1 to 
            inclusive of horizon_max using a step of -1, which 
            defines memory capacity task as a multi-output task, i.e., one 
            task per horizon
        win : int, optional
            initial window of the input signal to be used for generating the
            delayed output signal, by default 30
            note that horizon_max should exceed this window (in
            absolute value), otherwise ValueError is thrown
        low : float, optional
            lower boundary of the output interval
        high : float, optional
            upper boundary of the output interval
        input_gain : float, optional
            gain on the input signal, i.e., scalar multiplier, by default None
        add_bias : bool, optional
            decides whether bias is added to the input signal or not,
            by default False
        seed : int, optional
            seed for reproducible random generation, by default None

        Returns
        -------
        x, y : torch.Tensor
            input and output tensors with shapes (n_samples, n_features)
            and (n_samples, n_targets) respectively

        Raises
        ------
        ValueError
            if maximum horizon exceeds win (in absolute value)
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # Generate horizon range
        sign = -1 if horizon_max < 0 else 1
        horizon = list(range(sign, horizon_max + sign, sign))
        
        # Validate window size
        abs_horizon_max = abs(horizon_max)
        if win < abs_horizon_max:
            raise ValueError("Absolute maximum horizon must be <= window size")

        # Setup random generator 
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        # Generate input signal
        total_samples = self.n_trials + win + abs_horizon_max + 1
        x = (torch.rand((total_samples, 1), generator=generator) * (high - low)) + low

        # Create delayed outputs 
        y_slices = []
        for h in horizon:
            start = win + h
            end = -abs_horizon_max + h - 1 or None
            y_slices.append(x[start:end])
        y = torch.cat(y_slices, dim=1)

        # Trim input signal
        x = x[win:-abs_horizon_max - 1]

        # Apply input gain 
        if input_gain is not None:
            x *= input_gain

        # Add bias term 
        if add_bias:
            x = torch.cat([torch.ones_like(x[:, [0]]), x], dim=1)

        # Update attributes 
        self.n_features = x.size(1)
        self.n_targets = y.size(1)
        self.horizon_max = abs_horizon_max

        return x, y


def get_task_list(repository):
    """
    Returns list of tasks in repository

    Parameters
    ----------
    repository : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    repository = repository.lower()
    if repository == 'neurogym':
        return NEUROGYM_TASKS
    elif repository == 'reservoirpy':
        return RESERVOIRPY_TASKS
    elif repository == 'conn2res':
        return CONN2RES_TASKS