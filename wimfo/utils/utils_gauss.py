import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import LinAlgError, cholesky


def get_cov(data, t=1, detrend=True, ret_data=False):
    """
    Construct a time-lagged covariance matrix from input data.

    Parameters:
    - data (numpy.ndarray):      2D or 3D array of shape (variables, timepoints, trials).
    - t (int):                   Time lag.
    - detrend (bool):            Whether to demean and standardize the data.

    Returns:
    - (numpy.ndarray):             Time-lagged covariance matrix.
    """

    if detrend:
        data = demean(data)
    past = data[:, :-t]
    future = data[:, t:]
    if len(data.shape) == 3:
        past = past.reshape(past.shape[0], past.shape[1] * past.shape[2], order="F")
        future = future.reshape(
            future.shape[0], future.shape[1] * future.shape[2], order="F"
        )

    data = np.vstack([past, future])
    if ret_data:
        return data @ data.T / data.shape[1], data
    else:
        return data @ data.T / data.shape[1]


def tdmi_from_cov(cov, xdim=2, pointwise=False, data=None):
    """
    Compute the TDMI from the covariance matrix.

    Parameters:
    - cov (numpy.ndarray):       Covariance matrix of the past and future vectors.
    - xdim (int):                Number of variables.
    - pointwise (bool):          Whether to compute pointwise TDMI.
    - data (numpy.ndarray):      Data for pointwise calculation.

    Returns:
    - mi (float):                TDMI (in bits).
    """

    return MI(
        cov,
        list(range(xdim)),
        list(range(xdim, 2 * xdim)),
        pointwise=pointwise,
        data=data,
    )


def h(S):
    """
    Computes the differential entropy of a multivariate Gaussian.

    Parameters:
    S : ndarray
        Covariance matrix of the system (assumed to be positive definite).

    Returns:
    float: Half the log-determinant of the covariance matrix.
    """
    sign, logdet = slogdet(S)
    if sign <= 0:
        raise LinAlgError("Matrix is not positive definite.")
    return 0.5 * logdet


def demean(data):
    """
    Demean and standardise time series to mean 0 and unit variance.

    Parameters:
        data (numpy.ndarray):
            Multivariate time series, shape (n, T, m), where n is the number of channels,
            T is the number of time steps, m the number of trials.
    Returns:
        (numpy.ndarray):
            Standardised multivariate time series of shape (n, T, m).
    """
    n, T, m = data.shape if len(data.shape) == 3 else (*data.shape, 1)
    data2d = data.reshape(n, T * m, order="F")

    mean = np.mean(data2d, axis=1, keepdims=True)
    std = np.std(data2d, axis=1, keepdims=True, ddof=1)

    data_dem = (data2d - mean) / std
    return (
        data_dem.reshape(n, T, m, order="F")
        if len(data.shape) == 3
        else data_dem.reshape(n, T, order="F")
    )


def MI(cov, idx1, idx2, pointwise=False, data=None):
    """
    Compute mutual information between two sets of variables.

    Parameters:
        cov (numpy.ndarray): Covariance matrix.
        idx1 (list): Indices of the first set of variables.
        idx2 (list): Indices of the second set of variables.
        pointwise (bool): Whether to compute pointwise mutual information.
        data (numpy.ndarray): Data for pointwise calculation.

    Returns:
        float: Mutual information value in BITS.
    """
    if isinstance(idx1, int) or np.isscalar(idx1):
        idx1 = [idx1]
    if isinstance(idx2, int) or np.isscalar(idx2):
        idx2 = [idx2]
    full_cov = cov[np.ix_(idx1 + idx2, idx1 + idx2)]
    cov1 = cov[np.ix_(idx1, idx1)]
    cov2 = cov[np.ix_(idx2, idx2)]
    mi = (h(cov1) + h(cov2) - h(full_cov)) / np.log(2)

    if not pointwise:
        return mi
    else:
        if data is None:
            raise ValueError("Data must be provided for pointwise mutual information.")

        log_px = log_likelihood(data[idx1, :], cov1)
        log_py = log_likelihood(data[idx2, :], cov2)
        log_pxy = log_likelihood(data[idx1 + idx2, :], full_cov)

        pmi = (log_pxy - log_px - log_py) / np.log(2)
        # assert np.isclose(mi, np.mean(pmi)), "Pointwise MI mean does not match non-pointwise MI."

        return (pmi, mi)


def log_likelihood(data, cov):
    """
    Compute the log-likelihood of each sample in `data` under a multivariate Gaussian
    with covariance `cov` and zero mean.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (d, N) with d variables and N samples.
    cov : np.ndarray
        Covariance matrix of shape (d, d).

    Returns
    -------
    ll  : np.ndarray
        Array of shape (N,) with the log-likelihood of each sample.
    """
    d = data.shape[0]
    inv_cov = np.linalg.inv(cov)
    q = np.sum((inv_cov @ data) * data, axis=0)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite")
    ll = -0.5 * (d * np.log(2 * np.pi) + logdet + q)
    return ll


def get_whitening_matrix(covP, dm, dx, dy):
    """
    Reconstruct the exact block-structured whitening matrix W used by gpid's
    whiten(), such that W @ covP @ W.T = sig_mxy (the whitened covariance).

    The transform is:
        W = blkdiag(Sigma_M^{-1/2}, Sigma_{X|M}^{-1/2}, Sigma_{Y|M}^{-1/2})
    applied symmetrically (left and right), which is an orthogonal-like
    block-diagonal linear map.
    """
    import scipy.linalg as la

    sig_m = covP[:dm, :dm]
    sig_x = covP[dm : dm + dx, dm : dm + dx]
    sig_y = covP[dm + dx :, dm + dx :]
    sig_xm = covP[dm : dm + dx, :dm]  # cross-cov X,M
    sig_ym = covP[dm + dx :, :dm]  # cross-cov Y,M

    W_m = la.inv(la.sqrtm(sig_m).real)  # shape (dm, dm)

    sig_m_white = W_m @ sig_m @ W_m.T  # = I
    sig_xm_white = sig_xm @ W_m.T  # cross-cov after M-whitening
    sig_ym_white = sig_ym @ W_m.T

    sig_x__m = (
        sig_x - sig_xm_white @ sig_xm_white.T
    )  # conditional cov X|M (after M-whiten)
    W_x = la.inv(la.sqrtm(sig_x__m).real)  # shape (dx, dx)

    sig_y__m = sig_y - sig_ym_white @ sig_ym_white.T
    W_y = la.inv(la.sqrtm(sig_y__m).real)  # shape (dy, dy)

    W = la.block_diag(W_m, W_x, W_y)  # shape (dm+dx+dy, dm+dx+dy)

    return W


def unwhiten_cov(covP, covQ_white, dm, dx, dy, sig_mxy=None):
    """
    Transform covQ from whitened space back to original space.

    Parameters:
        covP (np.ndarray): Original covariance matrix, shape (dm+dx+dy, dm+dx+dy).
        covQ_white (np.ndarray): Covariance in whitened space, shape (dm+dx+dy, dm+dx+dy).
        dm (int): Dimension of target M.
        dx (int): Dimension of source X.
        dy (int): Dimension of source Y.
        sig_mxy (np.ndarray, optional): Whitened covP for sanity check.

    Returns:
        covQ (np.ndarray): Covariance in original space.
    """
    import scipy.linalg as la

    W = get_whitening_matrix(covP, dm, dx, dy)

    if sig_mxy is not None:
        assert np.allclose(
            W @ covP @ W.T, sig_mxy, atol=1e-8
        ), "Whitening matrix reconstruction failed"

    W_inv = la.block_diag(
        la.inv(W[:dm, :dm]),
        la.inv(W[dm : dm + dx, dm : dm + dx]),
        la.inv(W[dm + dx :, dm + dx :]),
    )
    covQ = W_inv @ covQ_white @ W_inv.T
    covQ = (covQ + covQ.T) / 2
    return covQ


def OTE_Gaussian(data, mu1, cov1, mu2, cov2):
    """
    Map some data ~N(mu1,cov1) to ~N(mu2,cov2) via the Optimal Transport Equation for two Gaussians.

    Parameters:
        data (np.ndarray): Data points to be transformed, shape (n_variables, n_samples).
        mu1 (np.ndarray): Mean of the source Gaussian, shape (n_variables,).
        cov1 (np.ndarray): Covariance of the source Gaussian, shape (n_variables, n_variables).
        mu2 (np.ndarray): Mean of the target Gaussian, shape (n_variables,).
        cov2 (np.ndarray): Covariance of the target Gaussian, shape (n_variables, n_variables).
    """
    assert (
        data.shape[0]
        == mu1.shape[0]
        == cov1.shape[0]
        == cov1.shape[1]
        == mu2.shape[0]
        == cov2.shape[0]
        == cov2.shape[1]
    ), "Dimension mismatch."
    # Compute the optimal transport map
    cov1_inv_sqrt = np.linalg.inv(cholesky(cov1, lower=True))
    cov2_sqrt = cholesky(cov2, lower=True)
    A = cov2_sqrt @ cov1_inv_sqrt
    b = mu2 - A @ mu1
    # Apply the transformation
    transformed_data = A @ data + b[:, np.newaxis]
    return transformed_data


def reconstruct_whitened_cov(sig, hx, hy):

    dx = hx.shape[0]
    dy = hy.shape[0]
    dm = hx.shape[1]

    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])

    HM = np.vstack((hx, hy))
    cov_xy = covxy__m + HM @ HM.T

    cov = np.block([[np.eye(dm), HM.T], [HM, cov_xy]])

    return cov


# Pointwise helpers
def get_pointwise_red(covP_sub, res_broja, dm, dx, dy, data_sub):
    """
    Given the result of exact_gauss_tilde_pid (res_broja) and the corresponding
    submatrix covP_sub of S, compute the pointwise redundancy.

    Parameters:
        covP_sub  : np.ndarray, submatrix of S in gpid ordering [target, S1, S2]
        res_broja : tuple returned by exact_gauss_tilde_pid(..., ret_t_sigt=True)
        dm        : int, dimension of target (M)
        dx        : int, dimension of source 1 (X)
        dy        : int, dimension of source 2 (Y)
        data_sub  : np.ndarray, shape (dm+dx+dy, N), data rows matching covP_sub

    Returns:
        pmi_red   : np.ndarray, shape (N,), pointwise redundancy in BITS
        red       : float, analytic redundancy in BITS (== res_broja[7])
    """
    from gpid.utils import whiten

    red = res_broja[7]  # scalar analytic redundancy in bits
    minQ = res_broja[-1]  # optimal sig in whitened space

    # Whitened covariance and channel parameters
    sig_mxy, hx, hy, hxy, sigxy = whiten(covP_sub, dm, dx, dy, ret_channel_params=True)

    # Reconstruct covQ in whitened space (only XY|M block changes)
    covQ_white = reconstruct_whitened_cov(minQ, hx, hy)

    # Unwhiten back to original space
    covQ = unwhiten_cov(covP_sub, covQ_white, dm, dx, dy, sig_mxy=sig_mxy)

    # OTE: map data from P-space to Q-space
    mapped_data = OTE_Gaussian(
        data_sub, np.zeros(covP_sub.shape[0]), covP_sub, np.zeros(covQ.shape[0]), covQ
    )

    # Local 0-based indices within [target, S1, S2]
    local_t = list(range(dm))
    local_s1 = list(range(dm, dm + dx))
    local_s2 = list(range(dm + dx, dm + dx + dy))

    # Three PMI terms under Q (all in BITS)
    pmi_s12, mi_s12 = MI(
        covQ, local_t, local_s1 + local_s2, pointwise=True, data=mapped_data
    )
    pmi_s1, mi_s1 = MI(covQ, local_t, local_s1, pointwise=True, data=mapped_data)
    pmi_s2, mi_s2 = MI(covQ, local_t, local_s2, pointwise=True, data=mapped_data)

    # Pointwise redundancy = pmi(T;S1) + pmi(T;S2) - pmi(T;S1,S2) under Q
    pmi_red = pmi_s1 + pmi_s2 - pmi_s12

    return pmi_red, red


def get_pointwise_union(covP, covQ, nx, ny, data, verbose=False):
    """
    Compute pointwise double-union information.

    Parameters:
        covP    : np.ndarray, original covariance (correlation space), shape (2n, 2n)
        covQ    : np.ndarray, optimised covariance (returned by double_union with ret_Q=True)
        nx      : int, dimension of first block (first variables)
        ny      : int, dimension of second block (second variables)
        data    : np.ndarray, shape (2n, N)

    Returns:
        pt_union : np.ndarray, shape (N,), pointwise union information in BITS
    """
    assert covP.shape == covQ.shape, "Covariance matrices must have the same shape."
    mapped_data = OTE_Gaussian(
        data, np.zeros(covP.shape[0]), covP, np.zeros(covQ.shape[0]), covQ
    )

    assert (
        covP.shape[0] == nx + ny
    ), "Covariance shape does not match expected dimensions."
    s = list(range(nx))  # sources
    t = list(range(nx, nx + ny))  # targets

    pt_union, _ = MI(covQ, t, s, pointwise=True, data=mapped_data)
    if verbose:
        print(f"Pointwise double union mean is {pt_union.mean():.6f} bits")
    return pt_union
