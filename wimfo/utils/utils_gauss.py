import numpy as np
from numpy.linalg import slogdet


def get_cov(data, t=1, detrend=True):
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

    return np.cov(np.vstack([past, future]))


def tdmi_from_cov(cov, xdim=2):
    """
    Compute the TDMI from the covariance matrix.

    Parameters:
    - cov (numpy.ndarray):       Covariance matrix of the past and future vectors.
    - xdim (int):                Number of variables.

    Returns:
    - mi (float):                TDMI (in bits).
    """
    Sx = cov[:xdim, :xdim]
    Sy = cov[xdim:, xdim:]
    mi = 0.5 * (slogdet(Sx)[1] + slogdet(Sy)[1] - slogdet(cov)[1]) / np.log(2)
    return mi


def demean(data):
    """
    Demean and standardise time series to mean 0 and unit variance.

    Parameters:
    - data (numpy.ndarray):     Multivariate time series, shape (n,_channels, n_timesteps, n_trials).
    Returns:
    - (numpy.ndarray):          Standardised multivariate time series of shape (n,_channels, n_timesteps, n_trials).
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
