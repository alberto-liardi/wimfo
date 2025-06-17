import numpy as np
from numpy.linalg import slogdet


def get_cov(data, t=1):
    """
    Construct a time-lagged covariance matrix from input data.

    Parameters:
    - data (numpy.ndarray):      2D array of shape (variables, timepoints).
    - t (int):                   Time lag.

    Returns:
    - numpy.ndarray:             Time-lagged covariance matrix.
    """

    n_vars, n_time = data.shape
    n_samples = n_time - t

    # stack the past and future vectors
    vecs = np.vstack([data[:, :n_samples], data[:, t:]])

    return np.cov(vecs)


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
