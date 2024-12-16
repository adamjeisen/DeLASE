import numpy as np
from scipy.spatial.distance import cdist
import torch
from tqdm.auto import tqdm

def compute_lyaps(Js, dt=1, k=None, verbose=False):
    T, n = Js.shape[0], Js.shape[1]
    old_Q = np.eye(n)
    if k is None:
        k = n
    old_Q = old_Q[:, :k]
    lexp = np.zeros(k)
    lexp_counts = np.zeros(k)
    for t in tqdm(range(T), disable=not verbose):
        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(Js[t], old_Q))
        # force diagonal of R to be positive
        # (if QR = A then also QLL'R = A with L' = L^-1)
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)
        old_Q = mat_Q
        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)

        # filter zeros in mat_R (would lead to -infs)
        idx = np.where(diag_R > 0)
        lexp_i = np.zeros(diag_R.shape, dtype="float32")
        lexp_i[idx] = np.log(diag_R[idx])
#         lexp_i[np.where(diag_R == 0)] = np.inf
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    
    return np.sort(np.divide(lexp, lexp_counts)*(1/dt))[::-1]

def compute_lyapunov_spectrum(jacobians, dt):
    """
    Computes the full Lyapunov spectrum using the QR method.

    Parameters:
        jacobians (list of np.ndarray): Sequence of Jacobian matrices.
        num_iterations (int): Number of iterations to compute the Lyapunov spectrum.
        dt (float): Time step between successive Jacobians.

    Returns:
        np.ndarray: The Lyapunov spectrum (array of exponents).
    """
    # Dimension of the system
    dim = jacobians[0].shape[0]
    num_iterations = len(jacobians)

    # Initialize a random orthonormal matrix Q
    Q = np.eye(dim)
    np.random.seed(0)  # For reproducibility
    Q = np.linalg.qr(np.random.randn(dim, dim))[0]

    # To store the cumulative logarithm of the stretching factors
    log_stretching_factors = np.zeros(dim)

    for i in range(num_iterations):
        # Multiply the current Jacobian by Q
        Z = jacobians[i % len(jacobians)] @ Q

        # Perform QR decomposition on Z
        Q, R = np.linalg.qr(Z)

        # Logarithm of the diagonal entries of R gives the stretching factors
        log_stretching_factors += np.log(np.abs(np.diag(R)))

    # Compute the mean logarithmic stretching factor for each exponent
    lyapunov_spectrum = (log_stretching_factors / num_iterations) / dt

    return lyapunov_spectrum

def rnn(t, x, W, tau, g):
    return (1/tau)*(-x + g*W @ np.tanh(x))

def rnn_jacobian(x, W, g, tau, dt, N, use_torch=False, device='cpu', dtype='torch.DoubleTensor'):
    if use_torch:
        I = torch.eye(x.shape[1]).type(dtype).to(device)
        if len(x.shape) == 1:
            return I + (dt/tau)*(-I + (g*W @ torch.diag(1 - torch.tanh(x)**2)))
        else:
            return I.unsqueeze(0) + (dt/tau)*(-I.unsqueeze(0) + (g*W*((1 - torch.tanh(x)**2).unsqueeze(1))))
    else:
        if len(x.shape) == 1:
            return np.eye(N) + (dt/tau)*(-np.eye(N) + (g*W @ np.diag(1 - np.tanh(x)**2)))
        else:
            print((1 - np.tanh(x)**2)[:, np.newaxis].shape)
            return np.eye(N)[np.newaxis] + (dt/tau)*(-np.eye(N)[np.newaxis] + (g*W*(1 - np.tanh(x)**2)[:, np.newaxis]))

def compute_lyaps_and_jacobians(x, W, g, tau, dt, N, k=None, use_torch=False, device='cpu', dtype='torch.DoubleTensor', verbose=False):
    T, n = x.shape[0], x.shape[1]
    old_Q = np.eye(n)
    if k is None:
        k = n
    old_Q = old_Q[:, :k]
    lexp = np.zeros(k)
    lexp_counts = np.zeros(k)
    for t in tqdm(range(T), disable=not verbose):
        J = rnn_jacobian(x[t], W, g, tau, dt, N, use_torch, device, dtype)
        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(J, old_Q))
        # force diagonal of R to be positive
        # (if QR = A then also QLL'R = A with L' = L^-1)
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
#         print(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)
        old_Q = mat_Q
        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)

        # filter zeros in mat_R (would lead to -infs)
        idx = np.where(diag_R > 0)
        lexp_i = np.zeros(diag_R.shape, dtype="float32")
        lexp_i[idx] = np.log(diag_R[idx])
#         lexp_i[np.where(diag_R == 0)] = np.inf
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    
    return np.divide(lexp, lexp_counts)*(1/dt)

def local_covariance(points, dist_matrix, r, i):
    """
    Compute the local covariance matrix for points within a distance r using precomputed distances.

    Parameters:
    points (numpy.ndarray): Array of shape (T, N) where T is the number of observations and N is the number of observables.
    dist_matrix (numpy.ndarray): Precomputed pairwise distance matrix of shape (T, T).
    r (float): Distance threshold for neighborhood.
    i (int): Index of the current point for which to compute the local covariance matrix.

    Returns:
    numpy.ndarray: Local covariance matrix.
    """
    # Select points within distance r
    neighbors = points[dist_matrix[i] < r]

    if len(neighbors) > 1:  # Need at least 2 points to compute covariance
        return np.cov(neighbors, rowvar=False)
    else:
        return None

def participation_ratio(cov_matrix):
    """
    Compute the participation ratio from a covariance matrix.

    Parameters:
    cov_matrix (numpy.ndarray): Covariance matrix.

    Returns:
    float: The participation ratio.
    """
    # print(np.linalg.det(cov_matrix))
    # if cov_matrix is None or np.linalg.det(cov_matrix) == 0:
    #     return None  # Undefined for singular matrix

    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_squared = np.sum(eigenvalues ** 2)

    if sum_lambda_squared == 0:
        return None  # Undefined if all eigenvalues are zero

    pr = (sum_lambda ** 2) / sum_lambda_squared
    return pr

def D_PR(points, r, dist_matrix=None,verbose=False):
    """
    Compute the scale-dependent participation ratio D_PR(r).

    Parameters:
    points (numpy.ndarray): Array of shape (T, N) where T is the number of observations and N is the number of observables.
    r (float): Distance threshold for neighborhood.

    Returns:
    float: The averaged participation ratio over all points.
    """
    T = points.shape[0]
    
    # Precompute the pairwise distance matrix
    if dist_matrix is None:
        dist_matrix = cdist(points, points)

    participation_ratios = []

    for i in tqdm(range(T), disable=not verbose):
        local_cov = local_covariance(points, dist_matrix, r, i)
        if local_cov is not None:
            pr = participation_ratio(local_cov)
            if pr is not None:
                participation_ratios.append(pr)

    return np.mean(participation_ratios) if participation_ratios else None