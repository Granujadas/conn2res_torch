# -*- coding: utf-8 -*-
"""
Functionality to train readout module using PyTorch
"""
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, r2_score

from . import utils
from . import performance


class ReadoutDataset(Dataset):
    """PyTorch Dataset for readout module"""
    def __init__(self, X, y, sample_weight=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
        if sample_weight is not None:
            self.sample_weight = torch.FloatTensor(sample_weight)
        else:
            self.sample_weight = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None and self.sample_weight is not None:
            return self.X[idx], self.y[idx], self.sample_weight[idx]
        elif self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class ReadoutModel(nn.Module):
    """Base PyTorch model for readout"""
    def __init__(self, input_dim, output_dim, model_type='regression'):
        super(ReadoutModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.model_type = model_type
        
    def forward(self, x):
        output = self.linear(x)
        if self.model_type == 'classification':
            output = torch.sigmoid(output)
        return output


class Readout:
    """
    Readout module implemented in PyTorch
    """
    def __init__(self, estimator=None, y=None, device='cpu'):
        """
        Initialize readout module

        Parameters
        ----------
        estimator : str or torch.nn.Module, optional
            Either a string specifying model type ('regression' or 'classification')
            or an actual PyTorch model, by default None
        y : array-like, optional
            Target values used to automatically select model type, by default None
        device : str, optional
            Device to run model on ('cpu' or 'cuda'), by default 'cpu'
        """
        self.device = device
        
        if estimator is not None and y is not None:
            raise ValueError("y must be None if estimator is provided")
        elif estimator is not None and y is None:
            self.model = estimator
        elif estimator is None and y is not None:
            self.model = y
        else:
            self.model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, arg):
        """
        Set model based on input

        Parameters
        ----------
        arg : str, torch.nn.Module, or array-like
            Either specifies model type, provides actual model, or provides target values
            to automatically select model type
        """
        if isinstance(arg, str):
            # Default to regression if not specified
            model_type = arg.lower()
            if model_type not in ['regression', 'classification']:
                model_type = 'regression'
            self._model_type = model_type
            self._model = None  # Will be initialized during training
        elif isinstance(arg, nn.Module):
            self._model = arg.to(self.device)
            if hasattr(arg, 'model_type'):
                self._model_type = arg.model_type
            else:
                self._model_type = 'regression'
        elif isinstance(arg, (list, np.ndarray, torch.Tensor)):
            self._model = select_model(arg)
            self._model_type = self._determine_model_type(arg)
        else:
            raise TypeError(
                "arg must be either a str specifying model type, "
                "an instance of a PyTorch model, "
                "or a list/numpy.ndarray/torch.Tensor of target values (y)"
            )

    def _determine_model_type(self, y):
        """Determine model type based on target values"""
        if isinstance(y, (list, tuple)):
            y = utils.concat(y)
        if isinstance(y, torch.Tensor):
            y = y.numpy()
            
        y = y.squeeze()
        if y.ndim == 1:
            unique_vals = np.unique(y)
            if len(unique_vals) == 2:
                return 'classification'
            elif y.dtype.kind in ['i', 'u']:
                return 'classification'
            else:
                return 'regression'
        else:  # multi-output
            return 'regression'

    def train(self, X, y, sample_weight=None, epochs=100, lr=0.01, batch_size=32):
        """
        Train the readout model

        Parameters
        ----------
        X : array-like
            Input features
        y : array-like
            Target values
        sample_weight : array-like, optional
            Sample weights, by default None
        epochs : int, optional
            Number of training epochs, by default 100
        lr : float, optional
            Learning rate, by default 0.01
        batch_size : int, optional
            Batch size, by default 32
        """
        # Check and convert inputs
        X, y = _check_xy_type(X, y)
        X = _check_x_dims(X)
        y = _check_y_dims(y)
        
        if isinstance(sample_weight, (list, tuple)):
            sample_weight = utils.concat(sample_weight)

        # Initialize model if not already done
        if self._model is None:
            input_dim = X.shape[1]
            output_dim = 1 if y.ndim == 1 else y.shape[1]
            self._model = ReadoutModel(input_dim, output_dim, self._model_type).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = ReadoutDataset(X, y, sample_weight)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss and optimizer
        if self._model_type == 'classification':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        
        # Training loop
        self._model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                if len(batch) == 3:  # with sample weights
                    x_batch, y_batch, weights_batch = batch
                    weights_batch = weights_batch.to(self.device).unsqueeze(1)
                else:
                    x_batch, y_batch = batch
                    weights_batch = None
                
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self._model(x_batch)
                
                if weights_batch is not None:
                    loss = (criterion(outputs, y_batch) * weights_batch).mean()
                else:
                    loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()

    def test(self, X, y, sample_weight=None, metric=None, batch_size=32, **kwargs):
        """
        Evaluate the readout model

        Parameters
        ----------
        X : array-like
            Input features
        y : array-like
            Target values
        sample_weight : array-like, optional
            Sample weights, by default None
        metric : str or tuple of str, optional
            Evaluation metric(s), by default None
        batch_size : int, optional
            Batch size, by default 32

        Returns
        -------
        dict
            Dictionary of metric scores
        """
        # Check and convert inputs
        X, y = _check_xy_type(X, y)
        X = _check_x_dims(X)
        y = _check_y_dims(y)
        
        if isinstance(sample_weight, (list, tuple)):
            sample_weight = utils.concat(sample_weight)

        # Create datasets and dataloaders
        test_dataset = ReadoutDataset(X, y, sample_weight)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Assign default metric if None
        if metric is None:
            metric = ('score',)
        elif isinstance(metric, str):
            metric = (metric,)
            
        # Get predictions
        self._model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:  # with sample weights
                    x_batch, y_batch, _ = batch
                else:
                    x_batch, y_batch = batch
                
                x_batch = x_batch.to(self.device)
                outputs = self._model(x_batch)
                all_outputs.append(outputs.cpu())
                all_targets.append(y_batch.cpu())
        
        y_pred = torch.cat(all_outputs).numpy()
        y_true = torch.cat(all_targets).numpy()
        
        if sample_weight is not None:
            sample_weight = torch.FloatTensor(sample_weight).numpy()
        
        # Calculate metrics
        scores = {}
        for m in metric:
            if m == 'score':
                if self._model_type == 'classification':
                    scores[m] = accuracy_score(y_true, y_pred > 0.5, sample_weight=sample_weight)
                else:
                    scores[m] = r2_score(y_true, y_pred, sample_weight=sample_weight)
            else:
                func = getattr(performance, m)
                scores[m] = func(y_true, y_pred, sample_weight=sample_weight, **kwargs)
        
        return scores

    def run_task(
        self, X, y, sample_weight=None, frac_train=0.7, metric=None,
        readout_modules=None, readout_nodes=None, **kwargs
    ):
        """
        Complete task of training and testing readout model

        Parameters
        ----------
        X : array-like
            Input features
        y : array-like
            Target values
        sample_weight : array-like, optional
            Sample weights, by default None
        frac_train : float, optional
            Fraction of data for training, by default 0.7
        metric : str or tuple of str, optional
            Evaluation metric(s), by default None
        readout_modules : array-like, optional
            Module assignments for nodes, by default None
        readout_nodes : array-like, optional
            Subset of nodes to use, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation results
        """
        # Get train-test split
        try:
            (x_train, x_test), (y_train, y_test) = X, y
        except ValueError as exc:
            if not utils.check(X, y):
                xy_names = [type(X).__name__, type(y).__name__]
                raise TypeError(
                    f"X is {xy_names[0]} and y is {xy_names[1]}. X and y must be the same type"
                ) from exc

            x_train, x_test, y_train, y_test = train_test_split(
                    X, y, frac_train=frac_train)

        # Define sample_weight
        if isinstance(sample_weight, (list, tuple)):
            sample_weight_train, sample_weight_test = sample_weight
        else:
            sample_weight_train, sample_weight_test = _get_sample_weight(
                (y_train, y_test), split_set=sample_weight
            )

        # Define set(s) of readout nodes
        if readout_modules is not None and readout_nodes is not None:
            raise ValueError(
                "Only one of readout_nodes or readout_modules must be passed"
            )
        elif readout_modules is not None and readout_nodes is None:
            readout_nodes, ids = get_readout_nodes(readout_modules)

        # Train and test model
        if readout_nodes is None:
            self.train(
                x_train, y_train, sample_weight_train, **kwargs
            )

            score = self.test(
                x_test, y_test, sample_weight_test, metric=metric, **kwargs
            )

            df_scores = pd.DataFrame(data=score, index=[0])

        elif isinstance(readout_nodes, (list, tuple, np.ndarray)):
            # Convert to arrays to enable indexing with readout_nodes
            x_train, y_train = _check_xy_type(x_train, y_train)
            x_test, y_test = _check_xy_type(x_test, y_test)

            # readout_nodes is an array of arrays
            if all(isinstance(i, (list, tuple, np.ndarray)) for i in readout_nodes):
                df_scores = []
                for i, readouts in zip(ids, readout_nodes):
                    self.train(
                        x_train[:, readouts], y_train,
                        sample_weight_train, **kwargs
                    )

                    score = self.test(
                        x_test[:, readouts], y_test,
                        sample_weight_test, metric=metric, **kwargs
                    )

                    df = pd.DataFrame(data=score, index=[0])
                    df['module'] = i
                    df['n_nodes'] = len(readouts)
                    df_scores.append(df[['module', 'n_nodes'] + list(metric)])

                df_scores = pd.concat(df_scores)

            # readout_nodes is a single array
            else:
                self.train(
                    x_train[:, readout_nodes], y_train,
                    sample_weight_train, **kwargs
                )

                score = self.test(
                    x_test[:, readout_nodes], y_test,
                    sample_weight_test, metric=metric, **kwargs
                )

                df_scores = pd.DataFrame(data=score, index=[0])

        return df_scores


def select_model(y):
    """Select appropriate PyTorch model based on target values"""
    if isinstance(y, (list, tuple)):
        y = utils.concat(y)
    if isinstance(y, torch.Tensor):
        y = y.numpy()
        
    y = y.squeeze()
    if y.ndim == 1:
        unique_vals = np.unique(y)
        if len(unique_vals) == 2 or y.dtype.kind in ['i', 'u']:
            return 'classification'
        else:
            return 'regression'
    else:  # multi-output
        return 'regression'


# Keep the following helper functions unchanged as they are mostly data processing
def train_test_split(*args, frac_train=0.7, n_train=None):
    """Split data into training and test sets"""
    argout = []
    for arg in args:
        if n_train is None and isinstance(arg, list):
            n_train = int(frac_train * len(arg))
        if n_train is None and isinstance(arg, np.ndarray):
            n_train = int(frac_train * arg.shape[0])
        argout.extend([arg[:n_train], arg[n_train:]])
    return tuple(argout)


def _check_xy_type(X, y):
    """Check and convert input types"""
    if X is not None and isinstance(X, (list, tuple)):
        X = utils.concat(X)
    if y is not None and isinstance(y, (list, tuple)):
        y = utils.concat(y)
    return X, y


def _check_x_dims(X):
    """Check and adjust input dimensions"""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if X.ndim == 1:
        return X[:, np.newaxis]
    else:
        return X


def _check_y_dims(y):
    """Check and adjust target dimensions"""
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    return y.squeeze()


def _get_sample_weight(y, split_set=None):
    """Get sample weights for training and testing"""
    try:
        y_train, y_test = y
    except ValueError:
        y_train = y
        y_test = y

    sample_weight_train, sample_weight_test = None, None

    if split_set == 'train':
        sample_weight_train = _sample_weight(y_train, split_set)
    elif split_set == 'test':
        sample_weight_test = _sample_weight(y_test, split_set)
    elif split_set == 'both':
        sample_weight_train = _sample_weight(y_train, split_set)
        sample_weight_test = _sample_weight(y_test, split_set)

    return sample_weight_train, sample_weight_test


def _sample_weight(y, split_set, seed=None):
    """Calculate sample weights"""
    baseline = _baseline(y)[-1]
    baseline_type = _baseline_class(y)

    if isinstance(y, (list, tuple)):
        sections = utils.get_sections(y)
        y = utils.concat(y)
        convert_to_list = True
    else:
        convert_to_list = False

    if isinstance(y, torch.Tensor):
        y = y.numpy()

    if y.ndim == 2:
        y = y[:, -1]

    sample_weight = np.ones_like(y).astype(float)
    if baseline_type == 'class1':
        sample_weight[y == baseline] = 0
    elif baseline_type == 'class2':
        sample_weight[y == baseline] = 0
        sample_weight = utils.split(sample_weight, sections)
        lens = int(np.mean(
            [len(np.where(i == 1)[0]) for i in sample_weight.copy() if len(set(i)) > 1]
            ))
        for i in range(len(sample_weight)):
            if all(sample_weight[i] == 0):
                sample_weight[i][-lens:] = 1
        sample_weight = utils.concat(sample_weight)
    elif baseline_type == 'class3':
        pass

    if split_set == 'train':
        rng = np.random.default_rng(seed=seed)
        idx = np.where(sample_weight == 0)[0]
        sample_weight[idx] = rng.rand((len(idx)))

    if convert_to_list:
        sample_weight = utils.split(sample_weight, sections)

    return sample_weight


def _baseline(y):
    """Get baseline values from target"""
    if isinstance(y, (list, tuple)):
        y = utils.concat(y)
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    y = _check_y_dims(y)

    if y.ndim == 1:
        n_targets = 1
        values, counts = np.unique(y, return_counts=True)
        baseline = values[counts == counts.max()]
    elif y.ndim == 2:
        n_targets = y.shape[1]
        baseline = []
        for target in range(n_targets):
            values, counts = np.unique(y[:, target], return_counts=True)
            baseline.extend(values[counts == counts.max()])

    if not len(baseline) == n_targets:
        warnings.warn("There is more than one baseline value per target")

    return baseline


def _baseline_class(y):
    """Classify baseline type"""
    baseline = _baseline(y)[-1]

    labels_per_trial = []
    for trial in y:
        if isinstance(trial, torch.Tensor):
            trial = trial.numpy()
        if trial.ndim == 2:
            trial = trial[:, -1]

        values, counts = np.unique(trial, return_counts=True)
        diff_from_baseline = np.setdiff1d(values, baseline)

        if diff_from_baseline.size == 0:
            labels_per_trial.append(baseline)
        elif diff_from_baseline.size == 1:
            labels_per_trial.append(diff_from_baseline[0])
        elif diff_from_baseline.size > 1:
            labels_per_trial.append(diff_from_baseline[-1])

    baseline_exists = True
    if len(np.unique(labels_per_trial)) == 1:
        baseline_exists = False

    baseline_included = False
    if baseline in labels_per_trial:
        baseline_included = True

    if baseline_exists and not baseline_included:
        baseline_type = 'class1'
    elif baseline_exists and baseline_included:
        baseline_type = 'class2'
    elif not baseline_exists:
        baseline_type = 'class3'

    return baseline_type


def get_readout_nodes(readout_modules):
    """Get readout nodes from module definitions"""
    if isinstance(readout_modules, (list, tuple, np.ndarray)):
        if all(isinstance(i, (list, tuple, np.ndarray)) for i in readout_modules):
            ids = list(range(len(readout_modules)))
            readout_nodes = list(module for module in readout_modules)
        else:
            ids = list(set(readout_modules))
            readout_nodes = list(
                np.where(np.array(readout_modules) == i)[0] for i in ids
            )
    elif isinstance(readout_modules, dict):
        ids = list(readout_modules.keys())
        readout_nodes = list(readout_modules.values())
    else:
        raise TypeError("")

    return readout_nodes, ids
