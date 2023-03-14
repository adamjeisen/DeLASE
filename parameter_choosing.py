import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from delase import DeLASE
from performance_metrics import compute_integrated_performance, get_autocorrel_funcs
from utils import *

class ParameterGrid:
    def __init__(self, window_vals=np.array([10000]), p_vals=None, matrix_size_vals=None, r_thresh_vals=np.array([0.3, 0.4, 0.5]), lamb_vals = np.array([0, 1e-12, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
                        reseed_vals=np.array([1, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])):
        self.window_vals = window_vals
        if p_vals is not None and matrix_size_vals is not None:
            raise ValueError("p_vals and matrix_size cannot be provided at the same time! Pick one please :)")
        if p_vals is None and matrix_size_vals is None:
            p_vals = np.array([20])
        self.p_vals = p_vals
        self.matrix_size_vals = matrix_size_vals
        self.r_thresh_vals = r_thresh_vals
        self.lamb_vals = lamb_vals
        self.reseed_vals = reseed_vals
    
        if self.p_vals is not None:
            self.total_combinations = len(window_vals)*len(p_vals)*len(r_thresh_vals)*len(lamb_vals)*len(reseed_vals)
        else:
            self.total_combinations = len(window_vals)*len(matrix_size_vals)*len(r_thresh_vals)*len(lamb_vals)*len(reseed_vals)

def compute_delase_chroots(delase, stability_max_freq=500, stability_max_unstable_freq=None):
    result = {}
    delase.compute_jacobians()
    delase.get_stability(max_freq=stability_max_freq, max_unstable_freq=stability_max_unstable_freq)
    result['stability_params'] = delase.stability_params
    result['stability_freqs'] = delase.stability_freqs
    if delase.use_torch:
        result['stability_params'] = result['stability_params'].cpu().numpy()
        result['stability_freqs'] = result['stability_freqs'].cpu().numpy()
    
    return result

def fit_and_test_delase(signal, test_signal, window, p, parameter_grid, dt, compute_ip=True, autocorrel_true=None, integrated_performance_kwargs={},
                        compute_chroots=True, stability_max_freq=500, stability_max_unstable_freq=125, use_torch=False, device=None, track_reseeds=False, iterator=None, message_queue=None, worker_num=None):
    results = []

    integrated_performance_args = dict(
        autocorrel_true=autocorrel_true, 
        reseed_vals=parameter_grid.reseed_vals, 
        iterator=iterator if track_reseeds else None, 
        message_queue=message_queue if track_reseeds else None, 
        worker_num=worker_num, 
        full_return=True,
    )
    integrated_performance_args = integrated_performance_args | integrated_performance_kwargs

    # -----------
    # Compute hankel matrix and SVD
    # -----------
    delase = DeLASE(signal, p=p, dt=dt, use_torch=use_torch, device=device)
    for r_thresh in parameter_grid.r_thresh_vals:
        for lamb in parameter_grid.lamb_vals:

            result = dict(
                window=window,
                p=p,
                r_thresh=r_thresh,
                lamb=lamb,
            )

            # -----------
            # Compute HAVOK DMD
            # -----------
            delase.compute_havok_dmd(r_thresh=r_thresh, lamb=lamb)

            # -----------
            # Compute integrated performance
            # -----------
            if compute_ip:
                ret_dict = compute_integrated_performance(delase, test_signal, **integrated_performance_args)
                result = result | ret_dict
            
            # -----------
            # Compute characteristic roots
            # -----------
            if compute_chroots:
                ret_dict = compute_delase_chroots(delase, stability_max_freq, stability_max_unstable_freq)
                result = result | ret_dict
            
            # -----------
            # Append result
            # -----------
            results.append(result)

            # -----------
            # send a message that the task is complete
            # -----------
            if not track_reseeds:
                if iterator is not None:
                    iterator.update()
                if message_queue is not None:
                    message_queue.put((worker_num, "task complete", "DEBUG"))
    
    return results

def parameter_search(train_signal, test_signal, parameter_grid=None, dt=1, compute_ip=True, integrated_performance_kwargs={},
                        compute_chroots=True, stability_max_freq=500, stability_max_unstable_freq=125, use_torch=False, device=None, verbose=False, track_reseeds=True):
    if parameter_grid is None:
        parameter_grid = ParameterGrid()
    
    train_signal = numpy_torch_conversion(train_signal, use_torch, device)
    test_signal = numpy_torch_conversion(test_signal, use_torch, device)
    if use_torch:
        device = train_signal.device

    results = []

    num_its = parameter_grid.total_combinations
    if not track_reseeds:
        num_its = int(num_its/len(parameter_grid.reseed_vals))
    iterator = tqdm(total=num_its, disable=not verbose)

    if 'num_lags' in integrated_performance_kwargs.keys():
        autocorrel_kwargs = {'num_lags': integrated_performance_kwargs['num_lags']}
    else:
        autocorrel_kwargs = {}
    autocorrel_true = get_autocorrel_funcs(test_signal, use_torch=use_torch, device=device, **autocorrel_kwargs)

    fit_and_test_args = dict(
        parameter_grid=parameter_grid, 
        dt=dt, 
        compute_ip=compute_ip, 
        autocorrel_true=autocorrel_true, 
        integrated_performance_kwargs=integrated_performance_kwargs, 
        compute_chroots=compute_chroots, 
        stability_max_freq=stability_max_freq, 
        stability_max_unstable_freq=stability_max_unstable_freq,
        use_torch=use_torch, 
        device=device, 
        iterator=iterator,
        track_reseeds=track_reseeds
    )

    if parameter_grid.p_vals is not None:
        for window in parameter_grid.window_vals:
            signal = train_signal[:window]
            for p in parameter_grid.p_vals:
                results.extend(fit_and_test_delase(signal, test_signal, window, p, **fit_and_test_args))
    else:
        for window in parameter_grid.window_vals:
            signal = train_signal[:window]
            for matrix_size in parameter_grid.matrix_size_vals:
                p = int(np.ceil(matrix_size/train_signal.shape[1]))
                results.extend(fit_and_test_delase(signal, test_signal, window, p, **fit_and_test_args))

    iterator.close()

    return pd.DataFrame(results)