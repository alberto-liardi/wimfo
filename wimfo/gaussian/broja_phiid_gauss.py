import numpy as np
from numpy.linalg import slogdet
from gpid.tilde_pid import exact_gauss_tilde_pid
from wimfo.gaussian.double_union_gauss import double_union as double_union_gauss
from wimfo.utils.phiid_lattice import get_lattice


def MI(S, s, t):
    """
    Given a joint covariance matrix S, calculate the mutual information
    between variables indexed by the lists s and t.
    """
    Hx = slogdet(S[np.ix_(s, s)])[1]
    Hy = slogdet(S[np.ix_(t, t)])[1]
    Hxy = slogdet(S[np.ix_(s + t, s + t)])[1]
    return 0.5 * (Hx + Hy - Hxy)


def broja_phiid(S, nx=1, ny=1, verbose=False, optimiser="Adam", options=None, **kwargs):
    """
    Perform the Broja-PhiID decomposition on a given covariance matrix S
    for multivariate sources.
    NB: covariance matrix should be ordered with past variables first, then future variables.

    Parameters:
    - S:            Covariance matrix (square matrix of size (2*(nx+ny), 2*(nx+ny)))
    - nx:           Dimension of x
    - ny:           Dimension of y
    - optimiser:    Optimizer to use for Broja double union calculation ("Adam" or "Newton")
    - verbose:      Whether to print debug information
    - options:      Dictionary of extra options for optimization
    - kwargs:       Additional keyword arguments for compatibility

    Returns:
    - Decomposed information atoms according to Broja-PhiID framework.
    """

    if options == None:
        options = {}

    assert S.shape[0] == S.shape[1], "Input covariance matrix is not square."
    assert S.shape[0] == 2 * (
        nx + ny
    ), "Covariance matrix size does not match expected dimensions."

    # Define indices for past sources and future targets (x,y->a,b)
    x = list(range(nx))
    y = list(range(nx, nx + ny))
    a = list(range(nx + ny, nx + ny + nx))
    b = list(range(nx + ny + nx, 2 * (nx + ny)))

    # Compute Mutual Informations
    Ixa = MI(S, x, a) / np.log(2)
    Ixb = MI(S, x, b) / np.log(2)
    Iya = MI(S, y, a) / np.log(2)
    Iyb = MI(S, y, b) / np.log(2)

    Ixya = MI(S, x + y, a) / np.log(2)
    Ixyb = MI(S, x + y, b) / np.log(2)
    Ixab = MI(S, x, a + b) / np.log(2)
    Iyab = MI(S, y, a + b) / np.log(2)

    Ixyab = MI(S, x + y, a + b) / np.log(2)

    # Compute PID redundancies using I_Broja
    # NB. [future first, then past variables] [information is calculated in BITS]
    # uix, uiy, ri, si = ret[-4:]
    _, _, Rxytab, _ = exact_gauss_tilde_pid(
        S[np.ix_(a + b + x + y, a + b + x + y)], dm=nx + ny, dx=nx, dy=ny
    )[-4:]
    _, _, Rxyta, _ = exact_gauss_tilde_pid(
        S[np.ix_(a + x + y, a + x + y)], dm=nx, dx=nx, dy=ny
    )[-4:]
    _, _, Rxytb, _ = exact_gauss_tilde_pid(
        S[np.ix_(b + x + y, b + x + y)], dm=ny, dx=nx, dy=ny
    )[-4:]
    _, _, Rabtxy, _ = exact_gauss_tilde_pid(S, dm=nx + ny, dx=nx, dy=ny)[-4:]
    _, _, Rabtx, _ = exact_gauss_tilde_pid(
        S[np.ix_(x + a + b, x + a + b)], dm=nx, dx=nx, dy=ny
    )[-4:]
    _, _, Rabty, _ = exact_gauss_tilde_pid(
        S[np.ix_(y + a + b, y + a + b)], dm=ny, dx=nx, dy=ny
    )[-4:]

    # Compute PhiID union information
    double_union = double_union_gauss(
        S, nx=nx + ny, optimiser=optimiser, options=options, verbose=verbose
    ) / np.log(2)

    if verbose:
        print(f"Broja double union is {double_union}")

    if np.isnan(double_union):
        # if verbose:
        print("Broja double union calculation failed.")
        labels = [
            "rtr",
            "rta",
            "rtb",
            "rts",
            "xtr",
            "xta",
            "xtb",
            "xts",
            "ytr",
            "yta",
            "ytb",
            "yts",
            "str",
            "sta",
            "stb",
            "sts",
        ]
        return {label: np.nan for label in labels}

    return get_lattice(
        Ixa,
        Ixb,
        Iya,
        Iyb,
        Ixya,
        Ixyb,
        Ixab,
        Iyab,
        Ixyab,
        Rxytab,
        Rxyta,
        Rxytb,
        Rabtxy,
        Rabtx,
        Rabty,
        double_union,
        False,
    )


if __name__ == "__main__":

    nx = 2
    ny = 2

    # P is a two-bit-copy
    A = 0.9 * np.eye(nx + ny)
    from scipy.linalg import solve_discrete_lyapunov

    cov_X = solve_discrete_lyapunov(A, np.eye(nx + ny))
    cov_P = np.block([[cov_X, cov_X @ (A.T)], [A @ cov_X, cov_X]])

    res = broja_phiid(cov_P, nx, ny, verbose=True)
    print(res)
