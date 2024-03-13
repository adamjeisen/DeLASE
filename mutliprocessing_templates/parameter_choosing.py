import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from delase import DeLASE
from DeLASE.metrics import compute_AIC, compute_integrated_performance, get_autocorrel_funcs
# from utils import *

class ParameterGrid:
    def __init__(self, window_vals=np.array([10000]), n_delays_vals=None, matrix_size_vals=None, r_vals=None, reseed=False, reseed_vals=np.array([1, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])):
        self.window_vals = window_vals
        if n_delays_vals is not None and matrix_size_vals is not None:
            raise ValueError("p_vals and matrix_size cannot be provided at the same time! Pick one please :)")
        if n_delays_vals is None and matrix_size_vals is None:
            n_delays_vals=np.array([10])
        self.n_delays_vals = n_delays_vals
        self.matrix_size_vals = matrix_size_vals
        self.r_vals = r_vals
        self.reseed = reseed
        self.reseed_vals = reseed_vals
    
        if self.n_delays_vals is not None:
            num_expansions = len(n_delays_vals)
            self.expansion_type = 'n_delay'
            self.expansion_vals = n_delays_vals
        else:
            num_expansions = len(matrix_size_vals)
            self.expansion_type = 'matrix_size'
            self.expansion_vals = matrix_size_vals
        
        if reseed:
            self.total_combinations = len(window_vals)*num_expansions*len(r_vals)*len(reseed_vals)
        else:
            self.total_combinations = len(window_vals)*num_expansions*len(r_vals)

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

def fit_and_test_delase(signal, test_signal, window, expansion_val, parameter_grid, dt, norm_aic=True, compute_ip=False, autocorrel_true=None, integrated_performance_kwargs={},
                        compute_chroots=True, stability_max_freq=500, stability_max_unstable_freq=125, save_jacobians=False, device='cpu', track_reseeds=False, iterator=None, message_queue=None, worker_num=None, verbose=False):
    results = []

    integrated_performance_args = dict(
        autocorrel_true=autocorrel_true, 
        reseed_vals=parameter_grid.reseed_vals if parameter_grid.reseed else None, 
        iterator=iterator if track_reseeds else None, 
        message_queue=message_queue if track_reseeds else None, 
        worker_num=worker_num, 
        full_return=True,
    )
    integrated_performance_args = integrated_performance_args | integrated_performance_kwargs

    # -----------
    # Compute hankel matrix and SVD
    # -----------
    delase_init_args = {
        parameter_grid.expansion_type: expansion_val,
        'dt': dt,
        'device': device,
    }
    if verbose:
        if message_queue is not None:
            message_queue.put((worker_num, "Computing SVD...", "DEBUG"))
        else:
            print("Computing SVD...")
    delase = DeLASE(signal, **delase_init_args)
    delase.DMD.compute_hankel()
    delase.DMD.compute_svd()
    if verbose:
        if message_queue is not None:
            message_queue.put((worker_num, "Now running over ranks...", "DEBUG"))
        else:
            print("SVD computed!")
            print("Now running over ranks...")
    for r in parameter_grid.r_vals:
        if r <= delase.n*delase.n_delays:
            result = {
                'window': window,
                parameter_grid.expansion_type: expansion_val,
                'rank': r,
            }
            if verbose:
                if message_queue is not None:
                    message_queue.put((worker_num, f"Now computing for rank {r}", "DEBUG"))
                else:
                    print(f"Now computing for rank {r}")

            # -----------
            # Compute HAVOK DMD
            # -----------
            if verbose:
                if message_queue is not None:
                    message_queue.put((worker_num, f"Computing least squares fit to HAVOK DMD...", "DEBUG"))
                else:
                    print(f"Computing least squares fit to HAVOK DMD...")
            delase.DMD.compute_havok_dmd(r)
            if verbose:
                if message_queue is not None:
                    message_queue.put((worker_num, f"HAVOK DMD complete!", "DEBUG"))
                else:
                    print(f"HAVOK DMD complete!")

            # -----------
            # Compute AIC
            # -----------
            result['AIC'] = compute_AIC(delase, test_signal, norm=norm_aic)

            # -----------
            # Compute integrated performance
            # -----------
            if compute_ip:
                if verbose:
                    if message_queue is not None:
                        message_queue.put((worker_num, f"Computing integrated performance...", "DEBUG"))
                    else:
                        print(f"Computing integrated performance...")
                ret_dict = compute_integrated_performance(delase, test_signal, **integrated_performance_args, verbose=verbose if message_queue is None else False)
                result = result | ret_dict
                if verbose:
                    if message_queue is not None:
                        message_queue.put((worker_num, "Integrated performance computed!", "DEBUG"))
                    else:
                        print("Integrated performance computed!")
            
            # -----------
            # Compute characteristic roots
            # -----------
            if compute_chroots:
                if verbose:
                    if message_queue is not None:
                        message_queue.put((worker_num, "Computing characteristic roots...", "DEBUG"))
                    else:
                        print("Computing characteristic roots...")
                ret_dict = compute_delase_chroots(delase, stability_max_freq, stability_max_unstable_freq)
                result = result | ret_dict
                if verbose:
                    if message_queue is not None:
                        message_queue.put((worker_num, "Characteristic roots computed!", "DEBUG"))
                    else:
                        print("Characteristic roots computed!")
            # -----------
            # Save Jacobians
            # -----------
            if save_jacobians:
                if not compute_chroots:
                    delase.compute_jacobians()
                result['Js'] = delase.Js

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
        else:
            if not track_reseeds:
                if iterator is not None:
                    iterator.update()
                if message_queue is not None:
                    message_queue.put((worker_num, "task complete", "DEBUG"))
            else:
                if iterator is not None:
                    iterator.update(len(parameter_grid.reseed_vals))
                if message_queue is not None:
                    for i in range(len(parameter_grid.reseed_vals)):
                        message_queue.put((worker_num, "task complete", "DEBUG"))
        
    return results

def parameter_search(train_signal, test_signal, parameter_grid=None, dt=1, compute_ip=False, integrated_performance_kwargs={},
                        compute_chroots=True, stability_max_freq=500, stability_max_unstable_freq=125, use_torch=False, device='cpu', dtype='torch.DoubleTensor', verbose=False, track_reseeds=False):
    if parameter_grid is None:
        parameter_grid = ParameterGrid()
    
    # train_signal = numpy_torch_conversion(train_signal, use_torch, device, dtype)
    # test_signal = numpy_torch_conversion(test_signal, use_torch, device, dtype)
    if use_torch:
        device = train_signal.device

    results = []

    num_its = parameter_grid.total_combinations
    if not track_reseeds:
        if parameter_grid.reseed:   
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

    for window in parameter_grid.window_vals:
        signal = train_signal[:window]
        for expansion_val in parameter_grid.expansion_vals:
            if parameter_grid.expansion_type == 'matrix_size':
                n_delays = int(np.ceil(expansion_val/train_signal.shape[1]))
            if n_delays*train_signal.shape[1] < window - n_delays:
                results.extend(fit_and_test_delase(signal, test_signal, window, expansion_val, **fit_and_test_args))
            else:
                if track_reseeds:
                    iterator.update(len(parameter_grid.reseed_vals))
                else:
                    iterator.update(1)

    iterator.close()

    results = pd.DataFrame(results)

    return results.set_index(['window', parameter_grid.expansion_type, 'r'])

# ============================================================
# PICKING BASED ON INTEGRATED PERFORMANCE
# ============================================================

class ParameterGridIP:
    def __init__(self, window_vals=np.array([10000]), p_vals=None, matrix_size_vals=None, r_vals=None, r_thresh_vals=None, explained_variance_vals=None, lamb_vals = np.array([0, 1e-12, 1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
                        reseed_vals=np.array([1, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])):
        self.window_vals = window_vals
        if p_vals is not None and matrix_size_vals is not None:
            raise ValueError("p_vals and matrix_size cannot be provided at the same time! Pick one please :)")
        if p_vals is None and matrix_size_vals is None:
            matrix_size_vals=np.array([3000])
        self.p_vals = p_vals
        self.matrix_size_vals = matrix_size_vals
        none_vars = (r_thresh_vals is None) + (r_vals is None) + (explained_variance_vals is None)
        if none_vars < 2:
            raise ValueError("More than one value was provided between r_vals, r_thresh_vals, and explained_variance_vals. Please provide only one of these, and ensure the others are None!")
        elif none_vars == 3:
            explained_variance_vals=np.array([0.99])
        self.r_vals = r_vals
        self.r_thresh_vals = r_thresh_vals
        self.explained_variance_vals = explained_variance_vals
        self.lamb_vals = lamb_vals
        self.reseed_vals = reseed_vals
    
        if self.p_vals is not None:
            num_expansions = len(p_vals)
            self.expansion_type = 'p'
            self.expansion_vals = p_vals
        else:
            num_expansions = len(matrix_size_vals)
            self.expansion_type = 'matrix_size'
            self.expansion_vals = matrix_size_vals
        
        if self.r_vals is not None:
            num_low_ranks = len(r_vals)
            self.low_rank_type = 'r'
            self.low_rank_vals = r_vals
        elif self.r_thresh_vals is not None:
            num_low_ranks = len(r_thresh_vals)
            self.low_rank_type = 'r_thresh'
            self.low_rank_vals = r_thresh_vals
        else:
            num_low_ranks = len(explained_variance_vals)
            self.low_rank_type = 'explained_variance'
            self.low_rank_vals = explained_variance_vals
        
        self.total_combinations = len(window_vals)*num_expansions*num_low_ranks*len(lamb_vals)*len(reseed_vals)

def fit_and_test_delaseIP(signal, test_signal, window, expansion_val, parameter_grid, dt, compute_ip=True, autocorrel_true=None, integrated_performance_kwargs={},
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
    delase_init_args = {
        parameter_grid.expansion_type: expansion_val,
        'dt': dt,
        'use_torch': use_torch,
        'device': device
    }
    delase = DeLASE(signal, **delase_init_args)
    for low_rank_val in parameter_grid.low_rank_vals:
        for lamb in parameter_grid.lamb_vals:

            result = {
                'window': window,
                parameter_grid.expansion_type: expansion_val,
                parameter_grid.low_rank_type: low_rank_val,
                'lamb': lamb,
            }

            # -----------
            # Compute HAVOK DMD
            # -----------
            delase_havok_dmd_args = {
                parameter_grid.low_rank_type: low_rank_val,
                'lamb': lamb
            }
            delase.compute_havok_dmd(**delase_havok_dmd_args)

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

def parameter_searchIP(train_signal, test_signal, parameter_grid=None, dt=1, compute_ip=True, integrated_performance_kwargs={},
                        compute_chroots=True, stability_max_freq=500, stability_max_unstable_freq=125, use_torch=False, device=None, verbose=False, track_reseeds=True):
    if parameter_grid is None:
        parameter_grid = ParameterGrid()
    
    # train_signal = numpy_torch_conversion(train_signal, use_torch, device)
    # test_signal = numpy_torch_conversion(test_signal, use_torch, device)
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


    for window in parameter_grid.window_vals:
        signal = train_signal[:window]
        for expansion_val in parameter_grid.expansion_vals:
            if parameter_grid.expansion_type == 'matrix_size':
                p = int(np.ceil(expansion_val/train_signal.shape[1]))
            if p*train_signal.shape[1] < window - p:
                results.extend(fit_and_test_delase(signal, test_signal, window, expansion_val, **fit_and_test_args))
            else:
                if track_reseeds:
                    iterator.update(len(parameter_grid.reseed_vals))
                else:
                    iterator.update(1)

    iterator.close()

    return pd.DataFrame(results)