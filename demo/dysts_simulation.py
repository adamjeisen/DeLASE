"""
Dynamical systems in Python

(M, T, D) or (T, D) convention for outputs

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

Adapted from https://github.com/williamgilpin/dysts
Huge shoutout to William Gilpin for a fantastic repo, check out his work!
"""


from dataclasses import dataclass, field, asdict
import warnings
import json
import collections
import os
from scipy.integrate import solve_ivp
from scipy.signal import resample
from tqdm.auto import tqdm
import gzip

# data_path = "/om2/user/eisenaj/code/CommunicationTransformer/data/dynamical_systems.json"
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/dynamical_systems.json")

import numpy as np

try:
    from numba import jit, njit

    #     from jax import jit
    #     njit = jit

    has_jit = True
except ModuleNotFoundError:
    import numpy as np

    has_jit = False
    # Define placeholder functions
    def jit(func):
        return func

    njit = jit

staticjit = lambda func: staticmethod(
    njit(func)
)  # Compose staticmethod and jit decorators


try:
    from sdeint import itoint
except ImportError:
    _has_sdeint = False
else:
    _has_sdeint = True


data_default = {'bifurcation_parameter': None,
                'citation': None,
                 'correlation_dimension': None,
                 'delay': False,
                 'description': None,
                 'dt': 0.001,
                 'embedding_dimension': 3,
                 'hamiltonian': False,
                 'initial_conditions': [0.1, 0.1, 0.1],
                 'kaplan_yorke_dimension': None,
                 'lyapunov_spectrum_estimated': None,
                 'maximum_lyapunov_estimated': None,
                 'multiscale_entropy': None,
                 'nonautonomous': False,
                 'parameters': {},
                 'period': 10,
                 'pesin_entropy': None,
                 'unbounded_indices': [],
                 'positive_only': False,
                 'vectorize': False
               }
    
def standardize_ts(a, scale=1.0):
    """Standardize an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero
    
    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        scale (float): the number of standard deviations by which to scale
    
    Returns:
        ts_scaled (ndarray): A standardized time series with the same shape as 
            the input
    """
#     if len(a.shape) == 1: a = a[:, None]
    stds = np.std(a, axis=-2, keepdims=True)
    stds[stds==0] = 1
    mean_val = np.mean(a, axis=-2, keepdims=True)
    scale_val = scale * stds
    ts_scaled = (a - mean_val)/scale_val
    return ts_scaled, mean_val, scale_val

def integrate_dyn(f, ic, tvals, noise=0, dtval=None, **kwargs):
    """
    Given the RHS of a dynamical system, integrate the system
    noise > 0 requires the Python library sdeint (assumes Brownian noise)
    
    Args:
        f (callable): The right hand side of a system of ODEs.
        ic (ndarray): the initial conditions
        noise_amp (float or iterable): The amplitude of the Langevin forcing term. If a 
            vector is passed, this will be different for each dynamical variable
        dtval (float): The starting integration timestep. This will be the exact timestep for 
            fixed-step integrators, or stochastic integration.
        kwargs (dict): Arguments passed to scipy.integrate.solve_ivp.
        
    Returns:
        sol (ndarray): The integrated trajectory
    """
    ic = np.array(ic)
    
    if np.isscalar(noise):
        if noise > 0:
            noise_flag = True
        else:
            noise_flag = False
    else:
        if np.sum(np.abs(noise)) > 0:
            noise_flag = True
        else:
            noise_flag = False

    if noise_flag:
        if not _has_sdeint:
            raise ImportError("Please install the package sdeint in order to integrate with noise.")
        gw = lambda y, t: noise * np.diag(ic)
        fw = lambda y, t: np.array(f(y, t))
        tvals_fine = np.linspace(np.min(tvals), np.max(tvals), int(np.ptp(tvals)/dtval))
        sol_fine = itoint(fw, gw, np.array(ic), tvals_fine).T
        sol = np.vstack([resample(item, len(tvals)) for item in sol_fine])
    else:
        #dt = np.median(np.diff(tvals))
        fc = lambda t, y : f(y, t)
        sol0 = solve_ivp(fc, [tvals[0], tvals[-1]], ic, t_eval=tvals, first_step=dtval, **kwargs)
        sol = sol0.y
        #sol = odeint(f, np.array(ic), tvals).T

    return sol

@dataclass(init=False)
class BaseDyn:
    """A base class for dynamical systems
    
    Attributes:
        name (str): The name of the system
        params (dict): The parameters of the system.
        random_state (int): The seed for the random number generator. Defaults to None
        
    Development:
        Add a function to look up additional metadata, if requested
    """

    name: str = None
    params: dict = field(default_factory=dict)
    random_state: int = None
    dt: float = None

    def __init__(self, **entries):
        self.name = self.__class__.__name__
        self.loaded_data = self._load_data()
        self.params = self.loaded_data["parameters"]
        self.params = {key: entries[key] if key in entries else self.params[key] for key in self.params}
        if 'random_state' in entries:
            self.random_state = entries['random_state']

        # Cast all parameter arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)

        ic_val = self.loaded_data["initial_conditions"]
        if not np.isscalar(ic_val):
            ic_val = np.array(ic_val)
        self.ic = ic_val
        np.random.seed(self.random_state)

        for key in self.loaded_data.keys():
            setattr(self, key, self.loaded_data[key])
        
        if 'dt' in entries:
            self.dt = entries['dt']

    def redo_init(self, loaded_data):
        self.loaded_data = loaded_data
        self.params = self.loaded_data["parameters"]

        # Cast all parameter arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)

        ic_val = self.loaded_data["initial_conditions"]
        if not np.isscalar(ic_val):
            ic_val = np.array(ic_val)
        self.ic = ic_val
        np.random.seed(self.random_state)

        for key in self.loaded_data.keys():
            setattr(self, key, self.loaded_data[key])
    
    def update_params(self):
        """
        Update all instance attributes to match the values stored in the 
        `params` field
        """
        for key in self.params.keys():
            setattr(self, key, self.params[key])
    
    def get_param_names(self):
        return sorted(self.params.keys())

    def _load_data(self):
        """Load data from a JSON file"""
        # with open(os.path.join(curr_path, "chaotic_attractors.json"), "r") as read_file:
        #     data = json.load(read_file)
        with open(self.data_path, "r") as read_file:
            data = json.load(read_file)
        try:
            return data[self.name]
        except KeyError:
            print(f"No metadata available for {self.name}")
            #return {"parameters": None}
            return data_default

    @staticmethod
    def _rhs(X, t):
        """The right-hand side of the dynamical system"""
        return X

    @staticmethod
    def _jac(X, t):
        """The Jacobian of the dynamical system"""
        return X

    @staticmethod
    def bound_trajectory(traj):
        """Bound a trajectory within a periodic domain"""
        return np.mod(traj, 2 * np.pi)

class DynSys(BaseDyn):
    """
    A continuous dynamical system base class, which loads and assigns parameter
    values from a file

    Attributes:
        kwargs (dict): A dictionary of keyword arguments passed to the base dynamical
            model class
    """

    def __init__(self, **kwargs):
        self.data_path = data_path
        super().__init__(**kwargs)
        self.dt = self.loaded_data["dt"] if self.dt is None else self.dt
        self.period = self.loaded_data["period"]
        self.mean_val = None
        self.std_val = None

    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        if self.vectorize:
            out = self._rhs(X.T, t, **self.params)
        else:
            out = self._rhs(*X.T, t, **self.params)
        return out
    
    # AE writing this
    def jac(self, X, t, length=None):
        """The Jacobian of the dynamical equation"""
        if len(X.shape) == 1:
            if self.vectorize:
                out = np.array(self._jac(X, t, **self.params))
                # out = np.array(self._jac(X, t, **param_dict))
            else:
                out = np.array(self._jac(*X, t, **self.params))
        elif len(X.shape) == 3:
            if length is None:
                length = X.shape[1]
            out = np.zeros((X.shape[0], length, X.shape[2], X.shape[2]))
            for i in range(X.shape[0]):
                for j, _t in enumerate(t):
                    out[i, j] = self.jac(X[i, j], _t, **self.params)
        else:
            raise NotImplementedError("Shapes other than (D,) or (B, T, D) not supported")
        return out

    def __call__(self, X, t):
        """Wrapper around right hand side"""
        return self.rhs(X, t)
    
    def make_trajectory(
        self,
        n_periods,
        method="Radau",
        resample=True,
        pts_per_period=100,
        return_times=False,
        standardize=False,
        # postprocess=True,
        noise=0.0,
        num_ics=1,
        traj_offset_sd=1,
        verbose=False, 
    ):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and initial conditions
        
        Args:
            n (int): the total number of trajectory points
            method (str): the integration method
            resample (bool): whether to resample trajectories to have matching dominant 
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
            postprocess (bool): Whether to apply coordinate conversions and other domain-specific 
                rescalings to the integration coordinates
            noise (float): The amount of stochasticity in the integrated dynamics. This would correspond
                to Brownian motion in the absence of any forcing.
        
        Returns:
            sol (ndarray): A T x D trajectory
            tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory
            
        """
        # n = n_periods * pts_per_period
        n = int(n_periods * (self.period / self.dt))
        tpts = np.arange(n) * self.dt
        np.random.seed(self.random_state)

        if resample:
            # tlim = (self.period) * (n / pts_per_period)
            n = n_periods * pts_per_period
            tlim = self.period*n_periods
            # upscale_factor = (tlim / self.dt) / n
            upscale_factor = (self.period) / (pts_per_period * self.dt)
            # if upscale_factor > 1e3:
            if upscale_factor > 1e4:
                # warnings.warn(
                #     f"Expect slowdown due to excessive integration required; scale factor {upscale_factor}"
                # )
                warnings.warn(
                    f"New simulation timescale is more than 10000 times the original dt; scale factor {upscale_factor}"
                )
            tpts = np.linspace(0, tlim, n)
             
        if num_ics > 1:
            if len(np.array(self.ic).shape) == 1:
                self.ic = np.vstack([self.ic, np.zeros((num_ics - 1, len(self.ic)))])
            else:
                if self.ic.shape[0] == num_ics:
                    pass
                elif self.ic.shape[0] < num_ics:
                    self.ic = np.vstack([self.ic, np.zeros((num_ics - self.ic.shape[0], len(self.ic)))])
                else: # self.ic.shape[0] > num_ics
                    self.ic = self.ic[:num_ics, :]
            # for i in range(1, num_ics)

        m = len(np.array(self.ic).shape)
        if m < 1:
            m = 1
        if m == 1:

            sol = np.expand_dims(integrate_dyn(
                self, self.ic, tpts, dtval=self.dt, method=method, noise=noise
            ).T, 0)
        else:
            sol = list()
            for i, ic in tqdm(enumerate(self.ic), disable=not verbose, total=len(self.ic)):
                traj = integrate_dyn(
                    self, ic, tpts, dtval=self.dt, method=method, noise=noise
                )
                check_complete = (traj.shape[-1] == len(tpts))
                if check_complete: 
                    sol.append(traj)
                else:
                    warnings.warn(f"Integration did not complete for initial condition {ic}, skipping this point")
                    pass
                
                # select the next initial condition so it's on the attractor
                low = int(len(tpts)/4)
                high = len(tpts)
                if i < len(self.ic) - 1:
                    selected_ind = np.random.randint(low, high)
                    # sol is returned as indices x time
                    self.ic[i + 1] = sol[-1][:, selected_ind] + np.random.randn(sol[-1].shape[0])*traj_offset_sd
                    if self.positive_only:
                        self.ic[i + 1] = np.abs(self.ic[i + 1])
                    if self.name == 'LorotkaVoltarneodo':
                        self.ic[i + 1, -2:] = np.abs(self.ic[i + 1, -2:])
            sol = np.transpose(np.array(sol), (0, 2, 1))

        # if hasattr(self, "_postprocessing") and postprocess:
        #     warnings.warn(
        #         "This system has at least one unbounded variable, which has been mapped to a bounded domain. Pass argument postprocess=False in order to generate trajectories from the raw system."
        #     )
        #     sol2 = np.moveaxis(sol, (-1, 0), (0, -1))
        #     sol = np.squeeze(
        #         np.moveaxis(np.dstack(self._postprocessing(*sol2)), (0, 1), (1, 0))
        #     )

        if standardize:
            sol, self.mean_val, self.scale_val = standardize_ts(sol)

        if return_times:
            return {'dt': tpts[1] - tpts[0], 'time': tpts, 'values': sol}
        else:
            return sol
        
class VanDerPol(DynSys):
    @staticjit
    def _rhs(x, y, t, mu, gamma):
        xdot = y
        ydot = mu * (1 - x ** 2) * y - x - gamma * y
        return xdot, ydot

    @staticjit
    def _jac(x, y, t, mu, gamma):
        row1 = [0, 1]
        row2 = [-2 * mu * x * y - 1, mu * (1 - x ** 2) - gamma]
        return [row1, row2]
    
class Rossler(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c):
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * x - c * z
        return xdot, ydot, zdot

    @staticjit
    def _jac(x, y, z, t, a, b, c):
        result = np.zeros((3, 3))
        result[0, :] = [0, -1, -1]
        result[1, :] = [1, a, 0]
        result[2, :] = [z, 0, x - c]
        return result