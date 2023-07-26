from copy import deepcopy
import numpy as np
import scipy
from scipy.integrate import trapezoid
from scipy.stats import pearsonr
import torch
from tqdm.auto import tqdm

from utils import *

# ============================================================
# AIC
# ============================================================

def compute_AIC(delase, test_signal, norm=False):
    N = (test_signal.shape[0] - delase.n_delays)*test_signal.shape[1]
    if isinstance(test_signal, np.ndarray):
        test_signal = torch.from_numpy(test_signal)
    test_signal = test_signal.to(delase.device)
    preds = delase.DMD.predict(test_signal)
    AIC = float(N*torch.log(((preds[delase.n_delays:] - test_signal[delase.n_delays:])**2).sum()/N) + 2*(delase.DMD.A_v.shape[0]*delase.DMD.A_v.shape[1] + 1))

    if norm:
        AIC /= N

    return AIC

# ============================================================
# METRIC FUNCTIONS
# ============================================================

# num_time_points x num_dimensions
def pearsonr(x, y, use_torch=False, device=None, dtype='torch.DoubleTensor'):
    x = numpy_torch_conversion(x, use_torch, device, dtype)
    y = numpy_torch_conversion(y, use_torch, device, dtype)
    if len(x.shape) == 1: # y.shape should also have length 1
        if use_torch:
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
        else:
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
    x = x.T
    y = y.T
    if use_torch:
        xm = x - x.mean(axis=1).unsqueeze(-1)
        ym = y - y.mean(axis=1).unsqueeze(-1)
        num = (xm*ym).sum(axis=1)
        denom = torch.norm(xm, dim=1, p=2)*torch.norm(ym, dim=1, p=2)
    else:
        xm = x - np.expand_dims(x.mean(axis=1), -1)
        ym = y - np.expand_dims(y.mean(axis=1), -1)
        num = (xm*ym).sum(axis=1)
        denom = np.linalg.norm(xm, axis=1, ord=2)*np.linalg.norm(ym, axis=1, ord=2)

    return num/denom

# num_time_points x num_dimensions
def r2_metric(target, preds, use_torch=False, device=None, dtype='torch.DoubleTensor'):
    target = numpy_torch_conversion(target, use_torch, device, dtype)
    preds = numpy_torch_conversion(preds, use_torch, device, dtype)
    if len(target.shape) == 1: # pred.shape should also have length 1
        if use_torch:
            target = target.unsqueeze(-1)
            preds = preds.unsqueeze(-1)
        else:
            target = np.expand_dims(target, -1)
            preds = np.expand_dims(preds, -1)
    if use_torch:
        target_mean = torch.mean(target, axis=0).unsqueeze(0)
        ss_tot = torch.sum((target - target_mean) ** 2, axis=0)
        ss_res = torch.sum((target - preds) ** 2, axis=0)
    else:
        target_mean = np.expand_dims(np.mean(target, axis=0), 0)
        ss_tot = np.sum((target - target_mean) ** 2, axis=0)
        ss_res = np.sum((target - preds) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    return r2

# num_time_points x num_dimensions
def get_autocorrel_funcs(signal_multi_dim, num_lags=500, use_torch=False, device=None, dtype='torch.DoubleTensor', verbose=False):
    signal_multi_dim = numpy_torch_conversion(signal_multi_dim, use_torch, device, dtype)
    if use_torch:
        autocorrel_funcs = torch.zeros(signal_multi_dim.shape[1], num_lags).type(dtype).to(device)
    else:
        autocorrel_funcs = np.zeros((signal_multi_dim.shape[1], num_lags))
    for lag in range(num_lags):
        autocorrel_funcs[:, lag] = pearsonr(signal_multi_dim[lag:], signal_multi_dim[:signal_multi_dim.shape[0] - lag], use_torch=use_torch, device=device, dtype=dtype)


    return autocorrel_funcs

# num_time_points x num_dimensions
def correlation_matrix(mat, use_torch=False, device=None, dtype='torch.DoubleTensor'):
    mat = numpy_torch_conversion(mat, use_torch, device, dtype)
    if use_torch:
        corrmat = torch.zeros(mat.shape[1], mat.shape[1])
    else:
        corrmat = np.zeros((mat.shape[1], mat.shape[1]))
    for i in range(mat.shape[1]):
        for j in range(i):
            corrmat[i, j] = pearsonr(mat[:, [i]], mat[:, [j]], use_torch=use_torch, device=device, dtype=dtype)
            corrmat[j, i] = corrmat[i, j]
    
    return corrmat

# ------------------------------------------------------------
# FUNCTIONS FOR COMPUTING KL DIVERGENCE (from Brenner et. al. 2022)
# ------------------------------------------------------------

def clean_from_outliers(prior, posterior, use_torch=False):
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    if use_torch:
        nonzeros = nonzeros.double() 
    outlier_ratio = (1 - nonzeros).mean()
    return prior, posterior, outlier_ratio

def eval_likelihood_gmm_for_diagonal_cov(z, mu, std, use_torch=False):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    precision = 1 / (std ** 2)
    if use_torch:
        vec=vec.float()
        precision = torch.diag_embed(precision).float()
        prec_vec = torch.einsum('zij,azj->azi', precision, vec)
        exponent = torch.einsum('abc,abc->ab', vec, prec_vec)
        sqrt_det_of_cov = torch.prod(std, dim=1)
        likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
        likelihood = likelihood.sum(dim=1) / T
    else:
        precision = np.array([np.diag(precision[i]) for i in range(precision.shape[0])])
        prec_vec = np.einsum('zij,azj->azi', precision, vec)
        exponent = np.einsum('abc,abc->ab', vec, prec_vec)
        sqrt_det_of_cov = np.prod(std, axis=1)
        likelihood = np.exp(-0.5 * exponent) / sqrt_det_of_cov
        likelihood = likelihood.sum(axis=1) / T

    return likelihood

## KLX Statespace
def calc_div_mc(mu_inf, cov_inf, mu_gen, cov_gen, mc_n=1000, use_torch=False, device=None, dtype='torch.DoubleTensor'):
    if use_torch:
        t = torch.randint(0, mu_inf.shape[0], (mc_n,)).to(device)

        std_inf = torch.sqrt(cov_inf).type(dtype).to(device)
        std_gen = torch.sqrt(cov_gen).type(dtype).to(device)
        z_sample = (mu_inf[t] + std_inf[t] * torch.randn(mu_inf[t].shape).type(dtype).to(device)).reshape((mc_n, 1, -1))
    else:
        t = np.random.randint(0, mu_inf.shape[0], (mc_n, ))

        std_inf = np.sqrt(cov_inf)
        std_gen = np.sqrt(cov_gen)

        z_sample = (mu_inf[t] + std_inf[t]*np.random.randn(mu_inf[t].shape)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen, use_torch)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, std_inf, use_torch)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior, use_torch)

    if use_torch:
        div_mc = torch.mean(torch.log(posterior) - torch.log(prior), dim=0)
    else:
        div_mc = np.mean(np.log(posterior) - np.log(prior), axis=0)
    return div_mc, outlier_ratio

#TODO: not sure if numpy is working, just set use_torch=True
def calc_div_from_data(data_true, data_pred, num_samples=1, mc_n=1000, symmetric=False, use_torch=True, device=None, dtype='torch.DoubleTensor'):
    mu_gen = data_pred
    data_true = numpy_torch_conversion(data_true, use_torch, device, dtype)
    mu_gen = numpy_torch_conversion(mu_gen, use_torch, device, dtype)
    
    time_steps = min(len(data_true), 10000)
    mu_inf= data_true[:time_steps]
    
    mu_gen=mu_gen[:time_steps]

    scaling = 1.
    if use_torch:
        cov_inf = torch.ones(data_true.shape[-1]).repeat(time_steps, 1).type(dtype).to(device)*scaling
        cov_gen = torch.ones(data_true.shape[-1]).repeat(time_steps, 1).type(dtype).to(device)*scaling
        # cov_inf = torch.cov(data_true.T)
        # cov_gen = torch.cov(mu_gen.T)
    else:
        cov_inf = np.ones((time_steps, data_true.shape[-1]))
        cov_gen = np.ones((time_steps, data_true.shape[-1]))
        # cov_inf = np.cov(data_true.T)
        # cov_gen = np.cov(mu_gen.T)
    
    div_mc = 0
    mu_PQ = (mu_gen + mu_inf)/2
    cov_PQ = (cov_gen + cov_inf)/2
    for num_sample in range(num_samples):
        # div_mc_sample, _ = calc_div_mc(mu_gen, cov_gen, mu_inf, cov_inf, mc_n, use_torch, device)
        div_mc1, _  = calc_div_mc(mu_inf, cov_inf, mu_gen, cov_gen, mc_n, use_torch, device, dtype)
        div_mc2, _  = calc_div_mc(mu_gen, cov_gen, mu_gen, cov_gen, mc_n, use_torch, device, dtype)

        div_mc += 1 / 2 * (div_mc1 + div_mc2)
        # div_mc += div_mc_sample
    div_mc /= num_samples 

    #scaling = 1
   # mu_inf = get_posterior_mean(model.rec_model, x)
    #cov_true = scaling * tc.ones_like(data_true)
   # mu_gen = get_prior_mean(model.gen_model, time_steps)
    #cov_gen = scaling * tc.ones_like(data_gen)

    #kl_mc, _ = calc_kl_mc(data_true, cov_true, data_gen, cov_gen)
    return div_mc

# ============================================================
# COMPUTING ALL SIGNAL METRICS
# ============================================================

def signal_metrics(true_signal, pred_signal, metrics='all', num_lags=500, min_freq=0, max_freq=500, fft_n=None, dt=1, spectrum_smoothing_width=0, standardize=False, autocorrel_true=None, use_torch=False, device=None, dtype='torch.DoubleTensor', return_components=False, verbose=False):
    true_signal = numpy_torch_conversion(true_signal, use_torch, device, dtype)
    pred_signal = numpy_torch_conversion(pred_signal, use_torch, device, dtype)

    if metrics=='all':
        metrics = [
            'correl',
            'mse',
            'r2',
            'autocorrel_correl',
            'autocorrel_mse',
            'autocorrel_r2',
            'fft_correl',
            'fft_mse',
            'fft_r2',
            'log_fft_correl',
            'log_fft_mse',
            'log_fft_r2',
            'correl_mat_correl',
            'correl_mat_mse',
            'correl_mat_r2',
            'div'
        ]
    metric_vals = {}
    metric_components = {}

    # ------------------------------
    # RAW SIGNAL METRICS
    # ------------------------------

    if 'correl' in metrics:
        if verbose:
            print('Computing raw signal correlation')
        metric_vals['correl'] = pearsonr(true_signal, pred_signal, use_torch, device, dtype).mean()
    if 'mse' in metrics:
        if verbose:
            print('Computing raw signal MSE')
        metric_vals['mse'] = ((true_signal - pred_signal)**2).mean()
    if 'r2' in metrics:
        if verbose:
            print('Computing raw signal R2')
        metric_vals['r2'] = r2_metric(true_signal, pred_signal, use_torch, device, dtype).mean()
    
    # ------------------------------
    # AUTOCORRELATION METRICS
    # ------------------------------

    autocorrel_metrics = ['autocorrel_correl', 'autocorrel_mse', 'autocorrel_r2']
    if len(set(metrics).intersection(set(autocorrel_metrics))) > 0:
        if autocorrel_true is None:
            autocorrel_true = get_autocorrel_funcs(true_signal, num_lags, use_torch=use_torch, device=device, dtype=dtype)
        autocorrel_pred = get_autocorrel_funcs(pred_signal, num_lags, use_torch=use_torch, device=device, dtype=dtype)
        if return_components:
            metric_components['autocorrel'] = dict(
                true=autocorrel_true,
                pred=autocorrel_pred
            )

    if 'autocorrel_correl' in metrics:
        if verbose:
            print('Computing autocorrelation correlation')
        metric_vals['autocorrel_correl'] = pearsonr(autocorrel_true.T, autocorrel_pred.T).mean()
    if 'autocorrel_mse' in metrics:
        if verbose:
            print('Computing raw signal MSE')
        metric_vals['autocorrel_mse'] = ((autocorrel_true - autocorrel_pred)**2).mean()
    if 'autocorrel_r2' in metrics:
        if verbose:
            print('Computing raw signal R2')
        metric_vals['autocorrel_r2'] = r2_metric(autocorrel_true.T, autocorrel_pred.T).mean()
    
    # ------------------------------
    # FFT METRICS
    # ------------------------------

    fft_metrics = ['fft_correl', 'fft_mse', 'fft_r2', 'log_fft_correl', 'log_fft_mse', 'log_fft_r2']
    if len(set(metrics).intersection(set(fft_metrics))) > 0:
        if fft_n is None:
            fft_n = true_signal.shape[0]
        if use_torch:
            fft_true = torch.abs(torch.fft.rfft(true_signal.T, n=fft_n))
            fft_pred = torch.abs(torch.fft.rfft(pred_signal.T, n=fft_n))
            freqs = torch.fft.rfftfreq(fft_n, d=dt)
        else:
            fft_true = np.abs(np.fft.rfft(true_signal.T, n=fft_n))
            fft_pred = np.abs(np.fft.rfft(pred_signal.T, n=fft_n))
            freqs = np.fft.rfftfreq(fft_n, d=dt)
        
        if use_torch:
            freqs = freqs.cpu().numpy()
            fft_true = fft_true.cpu().numpy()
            fft_pred = fft_pred.cpu().numpy()
        
        freq_inds = np.logical_and(freqs <= max_freq, freqs >= min_freq)
        fft_true = fft_true[:, freq_inds]
        fft_pred = fft_pred[:, freq_inds]

        df = freqs[1]
        
        if spectrum_smoothing_width > 0:
            for i in range(fft_true.shape[0]):
                fft_true[i] = scipy.ndimage.gaussian_filter1d(np.abs(fft_true[i]), spectrum_smoothing_width/df)
                fft_pred[i] = scipy.ndimage.gaussian_filter1d(np.abs(fft_pred[i]), spectrum_smoothing_width/df)
        if standardize:
            fft_true_ = deepcopy(fft_true)
            fft_pred_ = deepcopy(fft_pred)
            for i in range(fft_true.shape[0]):
                fft_true[i] = (fft_true[i] - fft_true[i].mean())/fft_true.std()
                fft_pred[i] = (fft_pred[i] - fft_pred[i].mean())/fft_pred.std()
        freqs = freqs[freq_inds]

        if return_components:
            metric_components['fft'] = dict(
                true=fft_true,
                pred=fft_pred,
                freqs=freqs
            )
    
    if 'fft_correl' in metrics:
        if verbose:
            print('Computing power spectra correlation')
        metric_vals['fft_correl'] = pearsonr(fft_true.T, fft_pred.T).mean()
    if 'fft_mse' in metrics:
        if verbose:
            print('Computing power spectra MSE')
        metric_vals['fft_mse'] = ((fft_true - fft_pred)**2).mean()
    if 'fft_r2' in metrics:
        if verbose:
            print('Computing power spectra R2')
        metric_vals['fft_r2'] = r2_metric(fft_true.T, fft_pred.T).mean()

    log_fft_metrics = ['log_fft_correl', 'log_fft_mse', 'log_fft_r2']

    if len(set(metrics).intersection(set(log_fft_metrics))) > 0:
        log_fft_true = 10*np.log10(fft_true_)
        log_fft_pred = 10*np.log10(fft_pred_)

        if standardize:
            for i in range(fft_true.shape[0]):
                log_fft_true[i] = (log_fft_true[i] - log_fft_true[i].mean())/log_fft_true.std()
                log_fft_pred[i] = (log_fft_pred[i] - log_fft_pred[i].mean())/log_fft_pred.std()

        if return_components:
            metric_components['log_fft'] = dict(
                true=log_fft_true,
                pred=log_fft_pred,
                freqs=freqs
            )
    
    if 'log_fft_correl' in metrics:
        if verbose:
            print('Computing log spectra correlation')
        metric_vals['log_fft_correl'] = pearsonr(log_fft_true.T, log_fft_pred.T).mean()
    if 'log_fft_mse' in metrics:
        if verbose:
            print('Computing log power spectra correlation')
        metric_vals['log_fft_mse'] = ((log_fft_true - log_fft_pred)**2).mean()
    if 'log_fft_r2' in metrics:
        if verbose:
            print('Computing log power spectra R2')
        metric_vals['log_fft_r2'] = r2_metric(log_fft_true.T, log_fft_pred.T).mean()

    # ------------------------------
    # CORRELATION MATRIX METRICS
    # ------------------------------

    correl_mat_metrics = ['correl_mat_correl', 'correl_mat_mse', 'correl_mat_r2']
    if len(set(metrics).intersection(set(correl_mat_metrics))) > 0:
        true_correl_mat = correlation_matrix(true_signal, use_torch, device, dtype)
        pred_correl_mat = correlation_matrix(pred_signal, use_torch, device, dtype)
        if return_components:
            metric_components['correl_mat'] = dict(
                true=true_correl_mat,
                pred=pred_correl_mat
            )

    if 'correl_mat_correl' in metrics:
        if verbose:
            print('Computing correlation matrix correlation')
        metric_vals['correl_mat_correl'] = pearsonr(true_correl_mat.flatten(), pred_correl_mat.flatten(), use_torch, device, dtype)
    if 'correl_mat_mse' in metrics:
        if verbose:
            print('Computing correlation matrix MSE')
        metric_vals['correl_mat_mse'] = ((true_correl_mat - pred_correl_mat)**2).mean()
    if 'correl_mat_r2' in metrics:
        if verbose:
            print('Computing correlation matrix R2')
        metric_vals['correl_mat_r2'] = r2_metric(true_correl_mat.flatten(), pred_correl_mat.flatten(), use_torch, device, dtype)

    # ------------------------------
    # KL DIVERGENCE METRICS
    # ------------------------------

    if 'div' in metrics:
        metric_vals['div'] = calc_div_from_data(true_signal, pred_signal, use_torch=True, device=device, dtype=dtype)

    if return_components:
        return metric_vals, metric_components
    else:
        return metric_vals

# ============================================================
# INTEGRATED PERFORMANCE
# ============================================================

#TODO: max_freq = 100?
def compute_integrated_performance(delase, test_signal, metrics=['autocorrel_correl', 'fft_correl', 'fft_r2'], weights='equal', num_lags=500, max_freq=200, fft_n=None, 
            reseed_vals=np.array([1, 5, 10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]), autocorrel_true=None, dims_to_analyze=None, iterator=None, message_queue=None, worker_num=None, verbose=False, full_return=False):
    if weights == 'equal':
        weights = np.ones(len(metrics)) / len(metrics)
    else:
        if len(weights) != len(metrics):
            raise ValueError('weights must have the same length as metrics!')

    if dims_to_analyze is None:
        dims_to_analyze = np.arange(test_signal.shape[1])
    
    test_signal = numpy_torch_conversion(test_signal, delase.use_torch, delase.device, delase.dtype)

    if autocorrel_true is None:
        autocorrel_true = get_autocorrel_funcs(test_signal[delase.p:, dims_to_analyze], num_lags, use_torch=delase.use_torch, device=delase.device, dtype=delase.dtype)
    if delase.use_torch:
        performance_curve = torch.zeros(len(reseed_vals))
    else:
        performance_curve = np.zeros(len(reseed_vals))
    
    all_metric_vals = []
    if iterator is None:
        iterator = tqdm(total=len(reseed_vals), disable=not verbose)
        close = True
    else:
        close = False
    for i, reseed in enumerate(reseed_vals):
        pred_signal = delase.predict_havok_dmd(test_signal, tail_bite=True, reseed=reseed, use_real_coords=True)

        metric_vals = signal_metrics(test_signal[delase.p:, dims_to_analyze], pred_signal[delase.p:, dims_to_analyze], metrics=metrics, num_lags=num_lags, max_freq=max_freq, fft_n=fft_n, dt=delase.dt, autocorrel_true=autocorrel_true, use_torch=delase.use_torch, device=delase.device)
        all_metric_vals.append(metric_vals)
        if delase. use_torch:
            performance_curve[i] = torch.from_numpy(np.array(list(metric_vals.values()))*weights).sum()
        else:
            performance_curve[i] = np.sum(np.array(list(metric_vals.values()))*weights)

        iterator.update()
        if message_queue is not None:
            message_queue.put((worker_num, "task complete", "DEBUG"))
    if close:
        iterator.close()
    ip = trapezoid(x=reseed_vals/reseed_vals.max(), y=performance_curve)

    if delase.use_torch:
        performance_curve = performance_curve.cpu().numpy()

    if full_return:
        return dict(
            ip=ip, 
            performance_curve=performance_curve, 
            reseed_vals=reseed_vals, 
            all_metric_vals=all_metric_vals
        )
    else:
        return ip