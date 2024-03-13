# CREDIT GIVEN TO:
# Breda, D., Maset, S., & Vermiglio, R. (2009). TRACE-DDE: a Tool for Robust Analysis and 
# Characteristic Equations for Delay Differential Equations. In J. J. Loiseau, W. Michiels,
# S.-I. Niculescu, & R. Sipahi (Eds.), Topics in Time Delay Systems: Analysis, 
# Algorithms and Control (pp. 145–155). Springer Berlin Heidelberg.
#
# Code adapted from MATLAB (and altered slightly)

import numpy as np
import numpy.matlib as matlib
import torch

# -----------------------------------
# STABILITY ESTIMATION
# -----------------------------------

def product_safe(x):
    x_abs_sort_inds = np.abs(x).argsort()
    x = x[x_abs_sort_inds]
    prod = 1
    for i in range(int(len(x)/2)):
        prod *= x[i]
        prod *= x[x.shape[0] - i - 1]
    if len(x) % 2 != 0:
        prod *= x[int(len(x)/2)]
    
    # print(prod)

    return prod

def elle(i, yy, xx):
    xx_trimmed = xx[np.where(xx != xx[i])[0]]

    return product_safe(np.divide(yy - xx_trimmed, xx[i] - xx_trimmed))

def cheb(N, T):    
    if N == 0:
        return 0, 1

    xx = np.cos(np.pi*np.arange(N + 1)/N)
    xx = T*(xx - 1)/2
    c = np.multiply(np.hstack([2, np.ones(N - 1), 2]), np.power(-1, np.arange(N + 1)))
    X = matlib.repmat(xx, N + 1, 1).T
    dX = X - X.T
    D = np.divide(np.outer(c, 1/c), dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D.T, axis=0))
    
    return D, xx

def compute_DDE_chroots(Js, dt, N=20, use_torch=False, device=None):
    m = Js.shape[1] # system dimension
    
    k = Js.shape[0] # num delays
    tau = torch.arange(1, Js.shape[0] + 1).to(device)*dt

    L = Js
    
    L0 = torch.zeros(Js[0].shape).to(device)
    T = float(torch.max(tau))
    AN = torch.zeros(m*(N+1), m*(N+1)).to(device)

    # faster without torch
    D, xx = cheb(N, T)
    D = torch.from_numpy(D).to(device)

    AN[m:, :] = torch.kron(D[1:], torch.eye(m).to(device))

    G_norm = []
    for i in range(N + 1):
        if i == 0:
            G = L0
        else:
            G = torch.zeros(m, m).to(device)

        for l in range(L.shape[0]):
            G += L[l]*elle(i, -float(tau[l]), xx)

        AN[:m, m*i:m*(i + 1)] = G

    eigvals = torch.linalg.eigvals(AN)

    return eigvals