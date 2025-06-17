import numpy as np
import torch
import parametrization_cookbook.torch as pc
from torchmin import minimize


def log1pexp(x):
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.relu(x)


def reals1d_to_chol(x):
    device = x.device
    n = int((8 * x.shape[-1] + 1) ** 0.5 / 2)
    assert (
        x.shape[-1] == n * (n + 1) // 2
    ), f"Incorect size. It does not exist n such as n*(n+1)/2=={x.shape[-1]}"
    other_slices = (slice(None),) * (len(x.shape) - 1)
    y = torch.zeros(x.shape[:-1] + (n, n), device=device, dtype=x.dtype)
    y[other_slices + (torch.arange(n, device=device),) * 2] = log1pexp(
        x[other_slices + (slice(None, n),)]
    )
    y[other_slices + tuple(torch.tril_indices(n, n, -1, device=device))] = x[
        other_slices + (slice(n, None),)
    ]

    y /= torch.sqrt(torch.arange(1, n + 1, device=device))[
        (None,) * (len(x.shape) - 2) + (slice(None), None)
    ]

    return y


def reals_to_half_sphere(x):
    device = x.device
    n = x.shape[-1]
    other_shape_and_one = x.shape[:-1] + (1,)
    other_slices = (slice(None),) * (len(x.shape) - 1)
    ksi = (
        torch.pi
        / 2
        * torch.tanh(x / 2 / torch.sqrt(2 * torch.arange(n, 0, -1, device=device) - 1))
    )
    return torch.concatenate(
        (torch.sin(ksi), torch.ones(other_shape_and_one, device=device)), axis=-1
    ) * torch.concatenate(
        (
            torch.ones(other_shape_and_one, device=device),
            torch.exp(torch.cumsum(torch.log(torch.cos(ksi)), axis=-1)),
        ),
        axis=-1,
    )


def reals_to_corr_matrix(x):
    n = int((8 * x.shape[-1] + 1) ** 0.5 / 2 + 1)
    assert n * (n - 1) // 2 == x.shape[-1]
    other_slices = (slice(None),) * (len(x.shape) - 1)
    y = torch.zeros(x.shape[:-1] + (n, n))
    y[other_slices + (0, 0)] = 1.0
    for i in range(1, n):
        y[other_slices + (i, slice(None, i + 1))] = reals_to_half_sphere(
            x[other_slices + (slice((i * (i - 1) // 2), (i + 1) * (i) // 2),)]
        )

    z = y @ y.transpose(-2, -1)
    sqrt_diag_z = z[other_slices + (torch.arange(n),) * 2] ** 0.5
    z = (
        z
        / sqrt_diag_z[other_slices + (slice(None), None)]
        / sqrt_diag_z[other_slices + (None, slice(None))]
    )
    return z


def reals_to_corr_chol(x):
    n = int((8 * x.shape[-1] + 1) ** 0.5 / 2 + 1)
    assert n * (n - 1) // 2 == x.shape[-1]
    other_slices = (slice(None),) * (len(x.shape) - 1)
    y = torch.zeros(x.shape[:-1] + (n, n))
    y[other_slices + (0, 0)] = 1.0
    for i in range(1, n):
        y[other_slices + (i, slice(None, i + 1))] = reals_to_half_sphere(
            x[other_slices + (slice((i * (i - 1) // 2), (i + 1) * (i) // 2),)]
        )
    return y


def reals_to_diag_chol(x, d):
    n = int((8 * x.shape[-1] + 1) ** 0.5 / 2 + 1)
    assert n * (n - 1) // 2 == x.shape[-1]
    other_slices = (slice(None),) * (len(x.shape) - 1)
    y = torch.zeros(x.shape[:-1] + (n, n))
    y[other_slices + (0, 0)] = torch.sqrt(1.0 - d[0])
    for i in range(1, n):
        y[other_slices + (i, slice(None, i + 1))] = reals_to_half_sphere(
            x[other_slices + (slice((i * (i - 1) // 2), (i + 1) * (i) // 2),)]
        ) * torch.sqrt(1.0 - d[i])
    return y


def mi_from_cov(S, xdim=2):
    # Mutual information in Nats
    Sx = S[:xdim, :xdim]
    Sy = S[xdim:, xdim:]
    mi = 0.5 * (torch.logdet(Sx) + torch.logdet(Sy) - torch.logdet(S))
    return mi


def block(args):
    rows = [torch.cat(a, dim=1) for a in args]
    return torch.cat(rows, dim=0)


def Corr2Params(S, nx, ny):
    rs = pc.MatrixCorrelation(dim=nx + ny).params_to_reals1d(S).double()

    n = int((8 * rs.shape[-1] + 1) ** 0.5 / 2)
    L = torch.zeros((n + 1, n + 1), dtype=torch.float64)
    tril_indices = torch.tril_indices(n, n)
    L[tril_indices[0] + 1, tril_indices[1]] = rs

    indx = torch.tril_indices(nx, nx, -1)
    indy = torch.tril_indices(ny, ny, -1)

    theta_x = L[:nx, :nx][indx[0], indx[1]]
    theta_y = L[nx:, nx:][indy[0], indy[1]]

    return theta_x, theta_y


def double_union(
    cov_P, nx=2, optimiser="Adam", options=None, verbose=False, switch_opt=True
):
    """
    Perform optimization to compute the double union information decomposition using the specified optimiser.

    Parameters:
    - cov_P (numpy.ndarray or torch.Tensor):    Covariance matrix of the variables.
    - nx (int, optional):                       Number of variables in the first set. Default is 2.
    - optimiser (str, optional):                Optimisation algorithm to use. Options are "Adam" or "Newton". Default is "Adam".
    - options (dict, optional):                 Dictionary of options for the optimiser. Default is None.
    - verbose (bool, optional):                 If True, prints additional information during optimization. Default is False.
    - switch_opt (bool, optional):              If True, will switch to the other optimiser if one fails to converge. Default is True.

    Returns:
    - float:                                    The union information.
    """

    assert isinstance(nx, int), "nx must be an integer."
    assert (
        2 <= nx < cov_P.shape[0]
    ), "nx must be at least 2 and strictly smaller than the size of the covariance matrix."
    assert cov_P.shape[0] == cov_P.shape[1], "Covariance matrix must be square."

    # Set the optimisation options for the optimiser
    def set_options(options, optimiser):
        if optimiser == "Newton":
            options.setdefault("custom_wolfe", True)
            expected_keys = {
                "lr",
                "max_iter",
                "line_search",
                "xtol",
                "normp",
                "tikhonov",
                "handle_npd",
                "callback",
                "disp",
                "return_all",
                "custom_wolfe",
            }
        elif optimiser == "Adam":
            expected_keys = {"atol", "rtol", "max_iter", "window_size"}
        else:
            raise ValueError(
                "Optimiser not supported! Choose between 'Adam' or 'Newton'."
            )

        filtered_opt = {k: v for k, v in options.items() if k in expected_keys}
        if verbose and filtered_opt.keys() != options.keys():
            print(
                f"Warning: The following option parameters were ignored: \
                  {set(options.keys()) - set(expected_keys)}"
            )

        return filtered_opt

    optim_options = set_options(options.copy() if options else {}, optimiser)

    # convert covariance to correlation matrix
    if isinstance(cov_P, np.ndarray):
        cov_P = torch.tensor(cov_P, requires_grad=False).double()
    D = torch.diag(1 / torch.sqrt(torch.diag(cov_P)))
    corr_P = D @ cov_P @ D
    cov_XY_P = corr_P[nx:, :nx].clone().detach().requires_grad_(False)
    ny = corr_P.shape[0] - nx

    try:
        torch.linalg.cholesky(corr_P)
    except:
        raise ValueError("Input covariance matrix is not positive definite.")

    # define loss function
    def loss(theta):
        Lxx = reals_to_corr_chol(theta[0, : nx * (nx - 1) // 2])
        Lyx = torch.linalg.solve_triangular(
            Lxx.transpose(0, 1), cov_XY_P, upper=True, left=False
        )
        d = torch.diag(Lyx @ Lyx.transpose(0, 1))
        Lyy = reals_to_diag_chol(theta[0, nx * (nx - 1) // 2 :], d)
        L_Q = block(((Lxx, torch.zeros((nx, ny), requires_grad=False)), (Lyx, Lyy)))
        cov_Q = L_Q @ (L_Q.transpose(0, 1))

        return mi_from_cov(cov_Q, xdim=nx)

    if verbose:
        print(f"Optimizing with {optimiser}...")

    # Run the optimisation
    def run_optimizer(opt, opt_params, verbose):
        # choose starting point (input correlation)
        theta = torch.concatenate(Corr2Params(corr_P, nx, ny)).unsqueeze(0)

        if opt == "Adam":
            return Adam_optim(theta, loss, **opt_params, verbose=verbose)
        elif opt == "Newton":
            try:
                result = minimize(
                    loss, theta, method="newton-exact", options=opt_params
                )
                if result.success:
                    return float(loss(result.x).detach())
            except Exception as e:
                if verbose:
                    print(f"Newton optimisation failed due to: {e}")
        return np.nan

    loss_val = run_optimizer(optimiser, optim_options, verbose)

    # If optimization fails and switch_opt is enabled, try the other optimiser
    if np.isnan(loss_val) and switch_opt:
        new_opt = "Newton" if optimiser == "Adam" else "Adam"
        if verbose:
            print(
                f"{optimiser} optimisation did not converge. Switching to {new_opt}..."
            )
        # Reset options for the new optimiser
        optim_options = set_options(options.copy() if options else {}, new_opt)
        # Run the optimisation again
        loss_val = run_optimizer(new_opt, optim_options, verbose)

    return loss_val


def Adam_optim(
    theta, loss, atol=1e-6, rtol=1e-6, max_iter=10000, window_size=11, verbose=False
):

    losses = []
    theta.requires_grad = True
    # optimizer = torch.optim.SGD([theta], lr=0\.1)
    optimizer = torch.optim.Adam([theta], lr=0.1)
    for n in range(max_iter):
        optimizer.zero_grad()
        l = loss(theta)
        l.backward()
        optimizer.step()

        losses.append(l.clone().detach())
        if torch.isnan(l):
            break
        elif n > window_size and torch.allclose(
            l, torch.tensor(losses[-window_size:-1]), atol=atol, rtol=rtol
        ):
            return float(l.detach())

    # If the optimisation failed, try to see if it converged for higher tolerances
    # Exit if the optimisation failed early
    if len(losses) < window_size:
        return np.nan
    # Check for increased tolerance
    while atol < 1e-1:
        for l in losses[window_size:]:
            if torch.isnan(l):
                break
            elif torch.allclose(
                l, torch.tensor(losses[-window_size:-1]), atol=atol, rtol=rtol
            ):
                if verbose:
                    print(f"Convergence reached with atol={atol} and rtol={rtol}.")
                return float(l.detach())
        atol *= 10
        rtol *= 10
    return np.nan


if __name__ == "__main__":
    nx = 2
    ny = 2
    # Random P
    corr_P = reals_to_corr_matrix(
        torch.rand(((nx + ny) * (nx + ny - 1) // 2,), requires_grad=False)
    )
    cov_XY_P = corr_P[nx:, :nx]

    # P is a two-bit-copy
    nx = 2
    A = 0.9 * np.eye(nx)
    from scipy.linalg import solve_discrete_lyapunov

    cov_X = solve_discrete_lyapunov(A, np.eye(nx))
    cov_P = np.block([[cov_X, cov_X @ (A.T)], [A @ cov_X, cov_X]])

    print(double_union(cov_P, nx=2, verbose=True))
