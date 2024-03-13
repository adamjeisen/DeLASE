import numpy as np
import torch

# ============================================================
# AIC
# ============================================================

def compute_AIC(delase, test_signal, norm=True):
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
# STANDALONE METRICS
# ============================================================

def aic(y_true, y_pred, k, norm=True):
    """
    Akaike Information Criterion
    """
    N = y_true.shape[0]*y_true.shape[1]
    if isinstance(y_true, np.ndarray):
        AIC = N*np.log(((y_true - y_pred)**2).sum()/N) + 2*(k + 1)
    else:
        AIC = N*torch.log(((y_true - y_pred)**2).sum()/N) + 2*(k + 1)

    if norm:
        AIC /= N
    
    return AIC


def mase(y_true, y_pred, y_train=None):
    """
    Mean Absolute Scaled Error. If the time series are multivariate, the first axis is
    assumed to be the time dimension.
    """    
    if y_train is None:
        y_train = y_true
    if len(y_true.shape) == 2:
        if isinstance(y_true, np.ndarray):
            return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[1:] - y_train[:-1]))
        else:
            return torch.mean(torch.abs(y_true - y_pred)) / torch.mean(torch.abs(y_true[1:] - y_train[:-1]))
    elif len(y_true.shape) == 3:
        if isinstance(y_true, np.ndarray):
            return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[:, 1:] - y_train[:, :-1]))
        else:
            return torch.mean(torch.abs(y_true - y_pred)) / torch.mean(torch.abs(y_true[:, 1:] - y_train[:, :-1]))
    else:
        raise ValueError("y_true must be 2 or 3 dimensional")


def mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    if isinstance(y_true, np.ndarray):
        return np.mean(np.square(y_true - y_pred))
    else:
        return torch.mean(torch.square(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    R2 Score
    """
    if isinstance(y_true, np.ndarray):
        return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    else:
        return 1 - torch.sum(torch.square(y_true - y_pred)) / torch.sum(torch.square(y_true - torch.mean(y_true)))