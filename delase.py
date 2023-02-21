import numpy as np
import torch

from stability_estimation import compute_DDE_chroots
from utils import *

def embed_signal(x, m, tau=1, use_torch=False, device=None):
    x = numpy_torch_conversion(x, use_torch, device)
    if use_torch:
        device = x.device
        embedding = torch.zeros((x.shape[0] - (m - 1)*tau, x.shape[1]*m)).to(device)
    else:
        embedding = np.zeros((x.shape[0] - (m - 1) * tau, x.shape[1]*m))
    for d in range(m):
        embedding[:, d*x.shape[1]:(d + 1)*x.shape[1]] = x[(m - 1 - d)*tau:x.shape[0] - d*tau]
    
    return embedding

class DeLASE:
    def __init__(self, data, p=None, tau=1, dt=None, svd=True, use_torch=False, device=None, verbose=False):
        self.window = data.shape[0]
        self.n = data.shape[1]
        self.p = p
        self.tau = tau
        self.use_torch = use_torch

        data = numpy_torch_conversion(data, use_torch, device)
        self.data = data
        if use_torch:
            self.device = data.device
        else:
            self.device = None
        
        if p is not None:
            self.compute_hankel(verbose=verbose)
            if svd:
                self.compute_svd(verbose=verbose)
            else:
                self.U = None
                self.S = None
                self.V = None
                self.S_mat = None
                self.S_mat_inv = None
        else:
            self.H = None
        
        self.r = None
        self.A_v = None
        self.A = None

        self.dt = dt
        self.Js = None

        self.N_time_bins = None
        self.chroots = None
        self.stability_params = None
        self.stability_freqs = None
        
    def compute_hankel(self, p=None, tau=None, verbose=False):
        if verbose:
            print("Computing Hankel matrix ...")
        if p is None and self.p is None:
            raise ValueError("Embedding dim p has not been provided.")
        elif p is not None:
            self.p = p
        else: # p is None and self.p is not None
            p = self.p
        
        if tau is None and self.tau is None:
            raise ValueError("Embedding dim p has not been provided.")
        elif tau is not None:
            self.tau = tau
        else: # p is None and self.p is not None
            tau = self.tau

        self.H = embed_signal(self.data, p, tau, use_torch=self.use_torch, device=self.device)
        
        if verbose:
            print("Hankel matrix computed!")
    
    def compute_svd(self, verbose=False):
        if self.H is None:
            raise ValueError("Hankel computation has not been done! Run compute_hankel first")

        if verbose:
            print("Computing SVD on Hankel matrix ...")
        
        # M = H.T

        if self.use_torch:
            U, S, Vh = torch.linalg.svd(self.H.T, full_matrices=False)
        else:
            U, S, Vh = np.linalg.svd(self.H.T, full_matrices=False)
        
        V = Vh.T
        self.U = U
        self.S = S
        self.V = V

        if self.use_torch:
            self.S_mat = torch.zeros(self.p*self.n, self.p*self.n).to(self.device)
            self.S_mat_inv = torch.zeros(self.p*self.n, self.p*self.n).to(self.device)
        else:
            self.S_mat = np.zeros((self.p*self.n, self.p*self.n))
            self.S_mat_inv = np.zeros((self.p*self.n, self.p*self.n))
        self.S_mat[np.arange(len(S)), np.arange(len(S))] = S
        self.S_mat_inv[np.arange(len(S)), np.arange(len(S))] = 1/S
        
        if verbose:
            print("SVD complete!")
    
    def compute_havok_dmd(self, r=None, r_thresh=0.25, lamb=0, rcond=1e-42, verbose=False):
        if self.U is None:
            raise ValueError("SVD computation has not been done! Run compute_svd before this function")
        if verbose:
            print("Computing least squares fits to HAVOK DMD ...")
        if r is None:
            if self.S[-1] > r_thresh:
                r = len(self.S)
            else:
                if self.use_torch:
                    r = torch.argmax(torch.arange(len(self.S), 0, -1).to(self.device)*(self.S < r_thresh))
                else:
                    r = np.argmax(np.arange(len(self.S), 0, -1)*(self.S < r_thresh))
        self.r = r
        
        if self.use_torch:
            # A = torch.linalg.lstsq(self.V[:-1, :r], self.V[1:, :r], rcond=rcond)[0].T
            # A = torch.linalg.lstsq(self.V[:-1, r].T @ self.V[:-1, :r] + lamb*torch.eye(r).to(self.device), self.V[:-1, :r].T@self.V[1:, :r], rcond=rcond)[0].T
            A = (torch.linalg.inv(self.V[:-1, :r].T @ self.V[:-1, :r] + lamb*torch.eye(r).to(self.device))@self.V[:-1, :r].T@self.V[1:, :r]).T
        else:
            A = (np.linalg.inv(self.V[:-1, :r].T @ self.V[:-1, :r] + lamb*np.eye(r))@self.V[:-1, :r].T@self.V[1:, :r]).T
        self.A_v = A

        self.A = self.U @ self.S_mat[:, :self.r] @ self.A_v @ self.S_mat_inv[:self.r] @ self.U.T

        if verbose:
            print("Least squares complete!")
    
    def predict_havok_dmd(self, test_data, tail_bite=False, reseed=None, full_return=False, verbose=False):
        test_data = numpy_torch_conversion(test_data, self.use_torch, self.device)
        H_test = embed_signal(test_data, self.p, self.tau, use_torch=self.use_torch, device=self.device)
        V_test = (self.S_mat_inv[:self.r] @ self.U.T @ H_test.T).T

        if tail_bite:
            if reseed is None:
                reseed = V_test.shape[0] + 1
            if self.use_torch:
                V_test_havok_dmd = torch.zeros(V_test.shape).to(self.device)
            else:
                V_test_havok_dmd = np.zeros(V_test.shape)
            V_test_havok_dmd[0] = V_test[0]
            for t in range(1, V_test.shape[0]):
                if t % reseed == 0:
                    V_test_havok_dmd[t] = self.A_v @ V_test[t - 1]
                else:
                    V_test_havok_dmd[t] = self.A_v @ V_test_havok_dmd[t - 1]
        else:
            V_test_havok_dmd_ = (self.A_v @ V_test[:-1].T).T
            if self.use_torch:
                V_test_havok_dmd = torch.vstack([V_test[[0], :self.r], V_test_havok_dmd_])
            else:
                V_test_havok_dmd = np.vstack([V_test[[0], :self.r], V_test_havok_dmd_])
        H_test_havok_dmd = (self.U @ self.S_mat[:, :self.r] @ V_test_havok_dmd.T).T

        if self.use_torch:
            pred_data = torch.vstack([test_data[:self.p], H_test_havok_dmd[1:, :self.n]])
        else:
            pred_data = np.vstack([test_data[:self.p], H_test_havok_dmd[1:, :self.n]])

        if full_return:
            return pred_data, H_test_havok_dmd, H_test, V_test_havok_dmd, V_test
        else:
            return pred_data
    
    def compute_jacobians(self, dt=None):
        if dt is None:
            if self.dt is None:
                raise ValueError("Time step dt required for computation!")
            else:
                dt = self.dt
        else:
            # overwrite saved dt
            self.dt = dt
        if self.use_torch:
            Js = torch.zeros(self.p, self.n, self.n).to(self.device)
        else:
            Js = np.zeros((self.p, self.n, self.n))
        for i in range(self.p):
            if i == 0:
                if self.use_torch:
                    Js[i] = (self.A[:self.n, i*self.n:(i + 1)*self.n] - torch.eye(self.n).to(self.device))/dt
                else:
                    Js[i] = (self.A[:self.n, i*self.n:(i + 1)*self.n] - np.eye(self.n))/dt
            else:
                Js[i] = self.A[:self.n, i*self.n:(i + 1)*self.n]/dt
        
        self.Js = Js

    def filter_chroots(self, max_freq=None):
        if self.use_torch:
            stability_params = torch.real(self.chroots)
            freqs = torch.imag(self.chroots)/(2*torch.pi)
        else:
            stability_params = np.real(self.chroots)
            freqs = np.imag(self.chroots)/(2*np.pi)

        if max_freq is not None:
            if self.use_torch:
                filtered_inds = torch.abs(freqs) <= max_freq
            else:
                filtered_inds = np.abs(freqs) <= max_freq
            stability_params = stability_params[filtered_inds]
            freqs = freqs[filtered_inds]

        self.stability_params = stability_params
        self.stability_freqs = freqs
    
    def get_stability(self, N_time_bins=None, max_freq=None):
        if self.Js is None:
            raise ValueError("Jacobians are needed for stability estimation! Run compute_jacobians first")

        if N_time_bins is None:
            N_time_bins = self.p
        self.N_time_bins = N_time_bins

        chroots = compute_DDE_chroots(self.Js, self.dt, N=N_time_bins, use_torch=self.use_torch, device=self.device)
        if self.use_torch:
            chroots = chroots[torch.argsort(torch.real(chroots)).flip(dims=(0,))]
        else:
            chroots = chroots[np.argsort(np.real(chroots))[::-1]]
        self.chroots = chroots

        self.filter_chroots(max_freq)