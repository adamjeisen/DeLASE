import numpy as np
import torch

from stability_estimation import compute_DDE_chroots
from utils import *

def embed_signal(x, m, tau=1, use_bias=False, use_torch=False, device=None, dtype='torch.DoubleTensor'):
    x = numpy_torch_conversion(x, use_torch, device, dtype)
    if use_torch:
        device = x.device
        embedding = torch.zeros((x.shape[0] - (m - 1)*tau, x.shape[1]*m)).type(dtype).to(device)
    else:
        embedding = np.zeros((x.shape[0] - (m - 1) * tau, x.shape[1]*m))
    for d in range(m):
        embedding[:, d*x.shape[1]:(d + 1)*x.shape[1]] = x[(m - 1 - d)*tau:x.shape[0] - d*tau]

    return embedding

class DeLASE:
    def __init__(self, data, p=None, matrix_size=None, tau=1, dt=None, svd=True, approx_rank=None, use_torch=False, device=None, dtype='torch.DoubleTensor', verbose=False):
        self.window = data.shape[0]
        self.n = data.shape[1]
        if p is not None and matrix_size is not None:
            raise ValueError("Cannot provide both p and matrix_size!!")
        elif p is None:
            self.matrix_size = matrix_size
            self.p = int(np.ceil(matrix_size/self.n))
        else: # matrix_size is None
            self.p = p
            self.matrix_size = self.n*self.p
        self.tau = tau
        self.dt = dt
        self.approx_rank = approx_rank
        self.use_torch = use_torch

        data = numpy_torch_conversion(data, use_torch, device, dtype)
        self.data = data
        if len(data.shape) == 3:
            self.trials = True
        else:
            self.trials = False
        if use_torch:
            self.device = data.device
            self.dtype = data.type()
        else:
            self.device = None
            self.dtype = None
        
        if self.p is not None:
            self.compute_hankel(verbose=verbose)
            if svd:
                self.compute_svd(verbose=verbose)
            else:
                self.U = None
                self.S = None
                self.V = None
                self.S_mat = None
                self.S_mat_inv = None
                self.cumulative_explained_variance = None
        else:
            self.H = None
        
        self.r = None
        self.scale_svd_coords = None
        self.V_coords = None
        self.use_bias = None
        self.A_v = None
        self.B_v = None
        self.A = None
        self.B = None

        
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

        if self.trials:
            if self.use_torch:
                H = torch.zeros(self.data.shape[0], self.data.shape[1] - p + 1, self.data.shape[2]*p).type(self.dtype).to(self.device)
            else:
                H = np.zeros(self.data.shape[0], self.data.shape[1] - p + 1, self.data.shape[2]*p)
            for i in range(self.data.shape[0]): 
                H[i] = embed_signal(self.data[i], p, use_torch=self.use_torch, device=self.device, dtype=self.dtype)
            self.H = H
        else:
            self.H = embed_signal(self.data, p, tau, use_torch=self.use_torch, device=self.device, dtype=self.dtype)
        
        if verbose:
            print("Hankel matrix computed!")
    
    def compute_svd(self, verbose=False):
        if self.H is None:
            raise ValueError("Hankel computation has not been done! Run compute_hankel first")

        if verbose:
            print("Computing SVD on Hankel matrix ...")
        
        if self.trials:
            H_in = self.H.reshape(self.H.shape[0]*self.H.shape[1], self.H.shape[2])
        else:
            H_in = self.H
        if self.approx_rank is not None:
            if self.use_torch:
                U, S, V = torch.svd_lowrank(H_in.T, self.approx_rank)
            else:
                raise ValueError("Can only approximate SVD in Torch!")
            
            self.U = U
            self.S = S
            self.V = V

            self.S_mat = torch.zeros(len(S), len(S)).type(self.dtype).to(self.device)
            self.S_mat_inv = torch.zeros(len(S), len(S)).type(self.dtype).to(self.device)
        else:
            if self.use_torch:
                U, S, Vh = torch.linalg.svd(H_in.T, full_matrices=False)
            else:
                U, S, Vh = np.linalg.svd(H_in.T, full_matrices=False)
            
            V = Vh.T
            self.U = U
            self.S = S
            self.V = V

            dim = np.min([H_in.shape[0], H_in.shape[1]])

            if self.use_torch:
                self.S_mat = torch.zeros(dim, dim).type(self.dtype).to(self.device)
                self.S_mat_inv = torch.zeros(dim, dim).type(self.dtype).to(self.device)
            else:
                self.S_mat = np.zeros((dim, dim))
                self.S_mat_inv = np.zeros((dim, dim))
        
        self.S_mat[np.arange(len(S)), np.arange(len(S))] = S
        self.S_mat_inv[np.arange(len(S)), np.arange(len(S))] = 1/S
    
        exp_variance_inds = self.S**2/((self.S**2).sum())
        if self.use_torch:
            cumulative_explained = torch.cumsum(exp_variance_inds, 0)
        else:
            cumulative_explained = np.cumsum(exp_variance_inds, 0)
        self.cumulative_explained_variance = cumulative_explained

        if verbose:
            print("SVD complete!")
    
    def compute_havok_dmd(self, r=None, r_thresh=None, explained_variance=None, lamb=0, use_bias=False, scale_svd_coords=False, rcond=1e-42, verbose=False):
        if self.U is None:
            raise ValueError("SVD computation has not been done! Run compute_svd before this function")
        if verbose:
            print("Computing least squares fits to HAVOK DMD ...")
        
        none_vars = (r is None) + (r_thresh is None) + (explained_variance is None)
        if none_vars < 2:
            raise ValueError("More than one value was provided between r, r_thresh, and explained_variance. Please provide only one of these, and ensure the others are None!")
        elif none_vars == 3:
            explained_variance=0.99

        if r_thresh is not None:
            if self.S[-1] > r_thresh:
                r = len(self.S)
            else:
                if self.use_torch:
                    r = torch.argmax(torch.arange(len(self.S), 0, -1).type(self.dtype).to(self.device)*(self.S < r_thresh))
                else:
                    r = np.argmax(np.arange(len(self.S), 0, -1)*(self.S < r_thresh))
        
        if explained_variance is not None:
            if self.use_torch:
                r = int(torch.argmax((self.cumulative_explained_variance > explained_variance).type(torch.int)).cpu().numpy())
            else:
                r = int(np.argmax((self.cumulative_explained_variance > explained_variance)))

        self.r = r

        self.scale_svd_coords = scale_svd_coords
        if scale_svd_coords:
            V_coord = ((self.S_mat @ self.V.T).T)[:, :r]
        else:
            V_coord = self.V[:, :r]

        self.use_bias = use_bias
        if use_bias:
            if self.use_torch:
                V_coord = torch.hstack([V_coord, torch.ones(V_coord.shape[0], 1).type(self.dtype).to(self.device)])
            else:
                V_coord = np.hstack([V_coord, np.ones((V_coord.shape[0], 1))])

        if self.trials:
            V_coord_trial = V_coord.reshape(self.data.shape[0], self.data.shape[1] - self.p + 1, V_coord.shape[1])
            V_minus = V_coord_trial[:, :-1].reshape(self.data.shape[0]*(self.data.shape[1] - self.p), V_coord.shape[1])
            V_plus = V_coord_trial[:, 1:].reshape(self.data.shape[0]*(self.data.shape[1] - self.p), V_coord.shape[1])
            del V_coord_trial
        
        else:
            V_minus = V_coord[:-1]
            V_plus = V_coord[1:]

        if self.use_torch:
            # A = torch.linalg.lstsq(self.V[:-1, :r], self.V[1:, :r], rcond=rcond)[0].T
            # A = torch.linalg.lstsq(self.V[:-1, r].T @ self.V[:-1, :r] + lamb*torch.eye(r).type(self.dtype).to(self.device), self.V[:-1, :r].T@self.V[1:, :r], rcond=rcond)[0].T
            # A = (torch.linalg.inv(self.V[:-1, :r].T @ self.V[:-1, :r] + lamb*torch.eye(r).type(self.dtype).to(self.device))@self.V[:-1, :r].T@self.V[1:, :r]).T
            A = (torch.linalg.inv(V_minus.T @ V_minus  + lamb*torch.eye(V_minus.shape[1]).type(self.dtype).to(self.device)) @ V_minus.T @ V_plus).T
        else:
            A = (np.linalg.inv(V_minus.T @ V_minus + lamb*np.eye(V_minus.shape[1])) @ V_minus.T @ V_plus).T
        if use_bias:
            self.A_v = A[:, :-1]
            self.B_v = A[:, -1]
        else:
            self.A_v = A

        if scale_svd_coords:
            self.A = self.U[:, :self.r] @ self.A_v @ self.U[:, :self.r].T
            if use_bias:
                self.B = self.U[:, :self.r] @ self.B_v
        else:
            self.A = self.U @ self.S_mat[:, :self.r] @ self.A_v @ self.S_mat_inv[:self.r] @ self.U.T
            if use_bias:
                self.B = self.U @ self.S_mat[:, :self.r] @ self.B_v

        if verbose:
            print("Least squares complete!")
    
    def predict_havok_dmd(self, test_data, tail_bite=False, reseed=None, use_real_coords=True, full_return=False, verbose=False):
        test_data = numpy_torch_conversion(test_data, self.use_torch, self.device, self.dtype)
        H_test = embed_signal(test_data, self.p, self.tau, use_torch=self.use_torch, device=self.device, dtype=self.dtype)
        if not use_real_coords:
            if self.scale_svd_coords:
                V_test = (self.U[:, :self.r].T @ H_test.T).T
            else:
                V_test = (self.S_mat_inv[:self.r] @ self.U.T @ H_test.T).T

        if tail_bite:
            if reseed is None:
                reseed = V_test.shape[0] + 1

            if use_real_coords:
                if self.use_torch:
                    H_test_havok_dmd = torch.zeros(H_test.shape).type(self.dtype).to(self.device)
                else:
                    H_test_havok_dmd = np.zeros(H_test.shape)
                H_test_havok_dmd[0] = H_test[0]

                for t in range(1, H_test.shape[0]):
                    if t % reseed == 0:
                        H_test_havok_dmd[t] = self.A @ H_test[t - 1]
                    else:
                        H_test_havok_dmd[t] = self.A @ H_test_havok_dmd[t - 1]
                    if self.use_bias:
                        H_test_havok_dmd[t] += self.B
            else: 
                if self.use_torch:
                    V_test_havok_dmd = torch.zeros(V_test.shape).type(self.dtype).to(self.device)
                else:
                    V_test_havok_dmd = np.zeros(V_test.shape)
                V_test_havok_dmd[0] = V_test[0]

                for t in range(1, V_test.shape[0]):
                    if t % reseed == 0:
                        V_test_havok_dmd[t] = self.A_v @ V_test[t - 1]
                    else:
                        V_test_havok_dmd[t] = self.A_v @ V_test_havok_dmd[t - 1]
                    if self.use_bias:
                        V_test_havok_dmd[t] += self.B_v
        else:
            if use_real_coords:
                H_test_havok_dmd_ = (self.A @ H_test[:-1].T).T
                if self.use_bias:
                    H_test_havok_dmd_ += self.B
                if self.use_torch:
                    H_test_havok_dmd = torch.vstack([H_test[[0]], H_test_havok_dmd_])
                else:
                    H_test_havok_dmd = np.vstack([H_test[[0]], H_test_havok_dmd_])
            else:
                V_test_havok_dmd_ = (self.A_v @ V_test[:-1].T).T
                if self.use_bias:
                    V_test_havok_dmd += self.B_v
                if self.use_torch:
                    V_test_havok_dmd = torch.vstack([V_test[[0], :self.r], V_test_havok_dmd_])
                else:
                    V_test_havok_dmd = np.vstack([V_test[[0], :self.r], V_test_havok_dmd_])
        
                if self.scale_svd_coords:
                    H_test_havok_dmd = (self.U[:, :self.r] @ V_test_havok_dmd.T).T
                else:
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
            Js = torch.zeros(self.p, self.n, self.n).type(self.dtype).to(self.device)
        else:
            Js = np.zeros((self.p, self.n, self.n))
        for i in range(self.p):
            if i == 0:
                if self.use_torch:
                    Js[i] = (self.A[:self.n, i*self.n:(i + 1)*self.n] - torch.eye(self.n).type(self.dtype).to(self.device))/dt
                else:
                    Js[i] = (self.A[:self.n, i*self.n:(i + 1)*self.n] - np.eye(self.n))/dt
            else:
                Js[i] = self.A[:self.n, i*self.n:(i + 1)*self.n]/dt
        
        self.Js = Js

    def filter_chroots(self, max_freq=None, max_unstable_freq=None):
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
        
        if max_unstable_freq is not None:
            if self.use_torch:
                filtered_inds = torch.logical_or(torch.abs(freqs) <= max_unstable_freq, stability_params <= 0)
            else:
                filtered_inds = np.logical_or(np.abs(freqs) <= max_unstable_freq, stability_params <= 0)
            stability_params = stability_params[filtered_inds]
            freqs = freqs[filtered_inds]

        self.stability_params = stability_params
        self.stability_freqs = freqs
    
    def get_stability(self, N_time_bins=None, max_freq=None, max_unstable_freq=None, sum_vals=False):
        self.compute_jacobians()
            # raise ValueError("Jacobians are needed for stability estimation! Run compute_jacobians first")

        if N_time_bins is None:
            N_time_bins = self.p
        self.N_time_bins = N_time_bins

        chroots = compute_DDE_chroots(self.Js, self.dt, N=N_time_bins, use_torch=self.use_torch, device=self.device, dtype=self.dtype, sum_vals=sum_vals)
        if self.use_torch:
            chroots = chroots[torch.argsort(torch.real(chroots)).flip(dims=(0,))]
        else:
            chroots = chroots[np.argsort(np.real(chroots))[::-1]]
        self.chroots = chroots

        self.filter_chroots(max_freq, max_unstable_freq)

    def to(self, device):
        if self.use_torch:
            self.device = device
            if self.data is not None:
                self.data = self.data.to(device)
            if self.H is not None:
                self.H = self.H.to(device)
            if self.U is not None:
                self.U = self.U.to(device)
            if self.S is not None:
                self.S = self.S.to(device)
            if self.V is not None:
                self.V = self.V.to(device)
            if self.S_mat is not None:
                self.S_mat = self.S_mat.to(device)
            if self.S_mat_inv is not None:
                self.S_mat_inv = self.S_mat_inv.to(device)
            if self.cumulative_explained_variance is not None:
                self.cumulative_explained_variance = self.cumulative_explained_variance.to(device)
            if self.V_coords is not None:
                self.V_coords = self.V_coords.to(device)
            if self.A_v is not None:
                self.A_v = self.A_v.to(device)
            if self.B_v is not None:
                self.B_v = self.B_v.to(device)
            if self.A is not None:
                self.A = self.A.to(device)
            if self.B is not None:
                self.B = self.B.to(device)
            if self.Js is not None:
                self.Js = self.Js.to(device)
            if self.chroots is not None:
                self.chroots = self.chroots.to(device)
            if self.stability_params is not None:
                self.stability_params = self.stability_params.to(device)
            if self.stability_freqs is not None:
                self.stability_freqs = self.stability_freqs.to(device)