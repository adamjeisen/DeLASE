from .metrics import aic, mase
from .delase import DeLASE
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def floor(x, eps=1e-8):
    if isinstance(x, torch.Tensor):
        return torch.floor(x + eps).type(torch.int64)
    else:
        return np.floor(x + eps).astype(int)

def perform_grid_search(data_sets, test_data_sets=None, dt=None, n_delays_vals=[1], rank_vals=None, delay_interval_vals=[1], max_freq=None, max_unstable_freq=None, compute_stability=True, device=None, verbose=False):
    """
    Perform a grid search over the hyperparameters of DeLASE.

    Parameters
    ----------
    data_sets : list of numpy.ndarray
        The data sets to perform the grid search on.
    test_data_sets : list of numpy.ndarray, optional
        The test data sets to perform the grid search on. If not provided, the training data sets will be used.
    dt : float, optional
        The time step size, used for computing the stability of the system.
    n_delays_vals : list of int, optional
        The number of delays to use for the grid search.
    rank_vals : list of int, optional
        The rank to use for the grid search.
    delay_interval_vals : list of int, optional
        The delay interval to use for the grid search.
    max_freq : float, optional
        The maximum frequency to use when considering characteristic roots computed in the stability analysis. 
    max_unstable_freq : float, optional
        The maximum frequency to use when considering unstable characteristic roots computed in the stability analysis.
        If not provided, max_freq will be used.
    compute_stability : bool, optional
        Whether to compute the stability of the system for each hyperparameter setting.
    device : str, optional
        The device to use for the grid search.
    verbose : bool, optional
        Whether to print progress to the console.
    
    Returns
    -------
    grid_search_results : list of pandas.DataFrame
        A list of dataframes, one for each data set, containing the results of the grid search.
    """
    if test_data_sets is None:
        test_data_sets = data_sets
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if rank_vals is None:
        rank_vals = [np.max([x.shape[-1] for x in data_sets])]
    
    if max_unstable_freq is None:
        max_unstable_freq = max_freq
    
    grid_search_results = []

    iterator = tqdm(total=len(data_sets)*len(n_delays_vals)*len(rank_vals)*len(delay_interval_vals), disable=not verbose, desc='Grid search')

    for x, x_test in zip(data_sets, test_data_sets):
        grid_search_results.append([])
        
        for i, n_delays in enumerate(n_delays_vals):
            for j, delay_interval in enumerate(delay_interval_vals):
                delase = DeLASE(x, n_delays=n_delays, rank=n_delays*x.shape[-1], delay_interval=delay_interval, dt=dt, max_freq=max_freq, max_unstable_freq=max_freq, device=device)
                delase.DMD.compute_hankel()
                delase.DMD.compute_svd()
                for k, rank in enumerate(rank_vals):
                    # only compute if the rank is less than the total number of dimensions
                    if rank <= n_delays*x.shape[-1]:
                        
                        delase.DMD.compute_havok_dmd(rank=rank)
                        preds = delase.DMD.predict(x_test)
                        aic_val = float(aic(torch.from_numpy(x_test).to(device)[..., (n_delays - 1)*delay_interval + 1:, :], preds[..., (n_delays - 1)*delay_interval + 1:, :], k=rank**2).cpu().numpy())
                        mase_val = float(mase(torch.from_numpy(x_test).cuda()[..., (n_delays - 1)*delay_interval + 1:, :], preds[..., (n_delays - 1)*delay_interval + 1:, :]).cpu().numpy())
                        if compute_stability:
                            delase.get_stability()
                            stability_params = delase.stability_params.cpu().numpy()
                    
                            num_params = np.min((floor(0.1*len(delase.stability_params)), rank))
                            num_params = np.max([num_params, 1])
                            stab_mean_top10percent = delase.stability_params[:num_params].mean().cpu().numpy()

                            stability_freqs = delase.stability_freqs.cpu().numpy()
                        else:
                            stability_params = None
                            stab_mean_top10percent = None
                            stability_freqs = None
                        
                        grid_search_results[-1].append(dict(
                            n_delays=n_delays,
                            rank=rank,
                            delay_interval=delay_interval,
                            aic=aic_val,
                            mase=mase_val,
                            stability_params=stability_params,
                            stab_mean_top10percent=stab_mean_top10percent,
                            stability_freqs=stability_freqs
                        ))
                    iterator.update()
        
        grid_search_results[-1] = pd.DataFrame(grid_search_results[-1])
    iterator.close()

    return grid_search_results

def get_results_matrix(grid_search_results, property, n_delays_vals, rank_vals, delay_interval_vals, top_percent = 0.1):
    """
    Get a results matrix for a given property.

    Parameters
    ----------
    grid_search_results : list of pandas.DataFrame
        The results of the grid search.
    property : str
        The property to get a results matrix for. Must be 'aic', 'mase', or 'stability'.
    n_delays_vals : list of int
        The number of delays to use for the grid search.
    rank_vals : list of int
        The rank to use for the grid search.
    delay_interval_vals : list of int
        The delay interval to use for the grid search.
    top_percent : float, optional
        The percentage of the top eigenvalues to use when computing the stability.

    Returns
    -------
    mat : numpy.ndarray
        A matrix of the results for the given property.
        The shape of the matrix is (num_trajs, len(n_delays_vals), len(rank_vals), len(delay_interval_vals)).
    """
    num_trajs = len(grid_search_results)
    if property not in ['aic', 'mase', 'stability']:
        raise ValueError(f"Invalid property: {property}, must be 'aic', 'mase', or 'stability'")

    if property == 'stability':
        if top_percent is None:
            raise ValueError("top_percent must be specified for stability")
        if top_percent <= 0 or top_percent >= 1:
            raise ValueError("top_percent must be between 0 and 1")
    
    mat = np.zeros((num_trajs, len(n_delays_vals), len(rank_vals), len(delay_interval_vals)))
    for num_traj in range(num_trajs):
        df = grid_search_results[num_traj]
        for i, n_delays in enumerate(n_delays_vals):
            for j, rank in enumerate(rank_vals):
                for k, delay_interval in enumerate(delay_interval_vals):
                        rows = df[(df.n_delays == n_delays) & (df['rank'] == rank) & (df['delay_interval'] == delay_interval)]
                        if len(rows) == 0:
                            mat[num_traj, i, j, k] = np.nan
                            continue
                        row = rows.iloc[0]
                        if property == 'stability':
                            num_params = int(np.round(top_percent*len(row.stability_params)))
                            num_params = np.max([num_params, 1])    
                            mat[num_traj, i, j, k] = row['stability_params'][:num_params].mean()
                        else:
                            mat[num_traj, i, j, k] = row[property]
    
    return mat

def get_optimal_hyperparameters(results_dict, n_delays_vals, rank_vals, delay_interval_vals):
    """
    Get the optimal hyperparameters for a given property.

    Parameters
    ----------
    results_dict : dict
        A dictionary where each key is a condition and each value is a results matrix output by get_results_matrix.
    n_delays_vals : list of int
        The number of delays to use for the grid search.
    rank_vals : list of int
        The rank to use for the grid search.
    delay_interval_vals : list of int
        The delay interval to use for the grid search.
    
    Returns
    -------
    optimal_hyperparameters : dict
        A dictionary containing the optimal hyperparameters for the given property.
    """
    # results_dict is a dictionary where each key is a condition and each value is a results matrix
    # first, let's concatenate all the results matrices
    all_results = np.concatenate([results_dict[key] for key in results_dict], axis=0)
    
    optimal_idx = np.unravel_index(np.nanargmin(np.nanmean(all_results, axis=(0,))), all_results.shape[1:])
    optimal_n_delays = n_delays_vals[optimal_idx[0]]
    optimal_rank = rank_vals[optimal_idx[1]]
    optimal_delay_interval = delay_interval_vals[optimal_idx[2]]

    optimal_hyperparameters = dict(
        n_delays=optimal_n_delays,
        rank=optimal_rank,
        delay_interval=optimal_delay_interval,
        idx=optimal_idx
    )

    return optimal_hyperparameters