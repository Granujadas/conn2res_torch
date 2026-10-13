import torch
from abc import abstractmethod
import torch.nn as nn
from torch.linalg import pinv
import torch.nn.functional as F

from conn2res import utils

class Reservoir(nn.Module):
    """
    Abstract base class for reservoir computing models.
    """
    def __init__(self, w):
        super().__init__()
        self.register_buffer('w', torch.as_tensor(w, dtype=torch.float32))
        self._state = None
        self.n_nodes = self.w.size(0)

    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass

    def add_washout_time(self, *args, idx_washout=0):
        """
        Remove initial transient dynamics (washout period).
        """
        return tuple(a[idx_washout:] for a in args)

class EchoStateNetwork(Reservoir):
    """
    Classic Echo State Network (ESN) implementation.
    """
    def __init__(self, w, activation_function='tanh', leak_rate=None, device='cpu'):
        super().__init__(w)
        self.device = device
        self.w = self.w.to(device)
        self.activation_function = self._set_activation_function(activation_function)
        self.leak_rate = leak_rate

    def simulate(self, ext_input, w_in, ic=None, output_nodes=None, return_states=True, **kwargs):
        """
        Simulate the ESN with external input and optional leakage.
        """
        if isinstance(ext_input, (list, tuple)):
            sections = utils.get_sections(ext_input)
            ext_input = utils.concat(ext_input)
            convert_to_list = True
        else:
            convert_to_list = False

        ext_input = torch.as_tensor(ext_input, device=self.device)
        w_in = torch.as_tensor(w_in, device=self.device)
        
        timesteps = ext_input.size(0)
        self._state = torch.zeros((timesteps + 1, self.n_nodes), device=self.device)

        if ic is not None:
            self._state[0] = torch.as_tensor(ic, device=self.device)

        for t in range(1, timesteps + 1):
            synap_input = self._state[t-1] @ self.w + ext_input[t-1] @ w_in
            if self.leak_rate is None:
                self._state[t] = self.activation_function(synap_input, **kwargs)
            else:
                self._state[t] = (1 - self.leak_rate) * self._state[t-1] + \
                                 self.leak_rate * self.activation_function(synap_input, **kwargs)

        self._state = self._state[1:]

        if return_states:
            if output_nodes is not None:
                return self._state[:, output_nodes]
            return self._state

    def _set_activation_function(self, function):
        """
        Select the activation function from common options.
        """
        activations = {
            'linear': lambda x, m=1: m * x,
            'elu': lambda x, alpha=0.5: F.elu(x, alpha),
            'relu': F.relu,
            'leaky_relu': lambda x, alpha=0.01: F.leaky_relu(x, alpha),
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'step': lambda x, thr=0.5: (x >= thr).float()
        }
        return activations[function]

class SpikingNeuralNetwork(Reservoir):
    """
    Placeholder for Spiking Neural Network reservoir (not implemented).
    """
    def __init__(self, w, inh=0.2, som=0.0, apply_Dale=True, device='cpu'):
        super().__init__(w)
        self.device = device
        self.w = self.w.to(device)

        if isinstance(inh, float):
            self.inh = torch.rand(self.n_nodes, device=device) < inh
        else:
            self.inh = torch.as_tensor(inh, dtype=torch.bool, device=device)

        self.exc = ~self.inh

        if isinstance(som, float):
            self.som = torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
            if som > 0:
                inh_idx = torch.where(self.inh)[0]
                som_size = int(som * len(inh_idx))
                self.som[inh_idx[:som_size]] = True
        else:
            self.som = torch.as_tensor(som, dtype=torch.bool, device=device)

        if apply_Dale and self.inh.any():
            mask = torch.eye(self.n_nodes, device=device)
            mask[self.inh] = -1
            self.w = torch.abs(self.w) @ mask

            som_mask = torch.ones_like(self.w)
            som_indices = torch.where(self.som)[0]
            som_mask[som_indices[:, None], self.inh] = 0
            self.w *= som_mask

    def simulate(self, *args, **kwargs):
        raise NotImplementedError("SpikingNeuralNetwork simulation is not yet implemented.")

class MemristiveReservoir(Reservoir):
    """
    Memristive reservoir model that solves internal voltages based on conductance matrix.
    """
    def __init__(self, w, int_nodes, ext_nodes, gr_nodes, save_conductance=False, device='cpu'):
        super().__init__(w)
        self.device = device
        self._I = torch.as_tensor(int_nodes, device=device)
        self._E = torch.as_tensor(ext_nodes, device=device)
        self._GR = torch.as_tensor(gr_nodes, device=device)
        self.save_conductance = save_conductance
        self._G_history = []

    def solveVi(self, Ve, Vgr=None, G=None):
        if Vgr is None:
            Vgr = torch.zeros(len(self._GR), device=self.device)
        if G is None:
            G = self._G

        N = torch.diag(G.sum(dim=1))
        A = N - G
        A_II = A[self._I][:, self._I]
        A_II_inv = pinv(A_II)

        H_IE = G[self._I][:, self._E] @ Ve
        H_IGR = G[self._I][:, self._GR] @ Vgr
        return A_II_inv @ (H_IE + H_IGR)

    def simulate(self, Vext, ic=None, mode='forward', dt=1e-4):
        Vext = torch.as_tensor(Vext, device=self.device)
        self._state = torch.zeros((len(Vext), self.n_nodes), device=self.device)

        for t, Ve in enumerate(Vext):
            if mode == 'forward':
                Vi = self.solveVi(Ve)
                V = self._get_voltages(Vi, Ve)
                self._update_conductance(V, dt)
            elif mode == 'backward':
                Vi = self._iterate(Ve)

            self._state[t, self._E] = Ve
            self._state[t, self._I] = Vi

            if self.save_conductance:
                self._G_history.append(self._G.clone())
        return self._state

    def _iterate(self, Ve, tol=5e-2, iters=100):
        """
        Iterative solver for internal voltages (backward mode).
        """
        Vi = torch.zeros(len(self._I), device=self.device)
        for _ in range(iters):
            V = self._get_voltages(Vi, Ve)
            self._update_conductance(V)
            new_Vi = self.solveVi(Ve)
            if torch.norm(new_Vi - Vi) < tol:
                break
            Vi = new_Vi
        return Vi

    def _get_voltages(self, Vi, Ve):
        """
        Combine internal and external voltages into a full voltage vector.
        """
        V = torch.zeros(self.n_nodes, device=self.device)
        V[self._I] = Vi
        V[self._E] = Ve
        return V

    def _update_conductance(self, V, dt=1e-4):
        """
        Dummy conductance update: identity for placeholder.
        """
        self._G = torch.eye(self.n_nodes, device=self.device)  # placeholder

class MSSNetwork(MemristiveReservoir):
    """
    Memristive network with stochastic switching.
    """
    def __init__(self, w, int_nodes, ext_nodes, gr_nodes, vA=0.17, vB=0.22, 
                 tc=0.32e-3, NMSS=10000, Woff=0.91e-3, Won=0.87e-2, Nb=2000, 
                 noise=0.1, device='cpu'):
        super().__init__(w, int_nodes, ext_nodes, gr_nodes, device=device)

        self.vA = self._init_property(vA, noise)
        self.vB = self._init_property(vB, noise)
        self.tc = self._init_property(tc, noise)
        self.NMSS = self._init_property(NMSS, noise)
        self.Woff = self._init_property(Woff, noise)
        self.Won = self._init_property(Won, noise)

        self._Ga = torch.where(self.NMSS != 0, self.Woff / self.NMSS, torch.zeros_like(self.Woff))
        self._Gb = torch.where(self.NMSS != 0, self.Won / self.NMSS, torch.zeros_like(self.Won))

        self._Nb = self._init_property(Nb, noise)
        self._G = self._Nb * (self._Gb - self._Ga) + self.NMSS * self._Ga

    def _init_property(self, mean, std=0.1):
        """
        Initialize property with Gaussian noise.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        noise = torch.randn_like(mean) * std
        return mean + noise

    def _update_conductance(self, V, dt=1e-4):
        """
        Placeholder for conductance update logic in MSS.
        """
        self._G = torch.clamp(self._G, min=1e-5, max=1.0)  # dummy clamping for safety