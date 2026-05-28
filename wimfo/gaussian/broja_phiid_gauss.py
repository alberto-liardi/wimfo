# broja_phiid.py
import numpy as np
from numpy.linalg import slogdet
from gpid.tilde_pid import exact_gauss_tilde_pid
from wimfo.gaussian.double_union_gauss import double_union as double_union_gauss
from wimfo.utils.phiid_lattice import get_lattice
from wimfo.utils.utils_gauss import MI, get_pointwise_union, get_pointwise_red


def broja_phiid(
    S,
    nx=1,
    ny=1,
    verbose=False,
    optimiser="Adam",
    options=None,
    pointwise=False,
    data=None,
    **kwargs,
):
    """
    Perform the Broja-PhiID decomposition on a given covariance matrix S.
    NB: covariance matrix should be ordered with past variables first, then future variables.

    Parameters:
        S          : np.ndarray, covariance matrix of shape (2*(nx+ny), 2*(nx+ny))
        nx         : int, dimension of x (past source 1)
        ny         : int, dimension of y (past source 2)
        verbose    : bool
        optimiser  : str, "Adam" or "Newton"
        options    : dict, extra options for optimiser
        pointwise  : bool, whether to compute pointwise decomposition
        data       : np.ndarray, shape (2*(nx+ny), N), required if pointwise=True

    Returns:
        If pointwise=False: dict of scalar atoms
        If pointwise=True:  [dict_of_arrays, dict_of_scalars]
    """

    if options is None:
        options = {}

    assert S.shape[0] == S.shape[1], "Input covariance matrix is not square."
    assert S.shape[0] == 2 * (
        nx + ny
    ), "Covariance matrix size does not match expected dimensions."

    if pointwise:
        assert data is not None, "Data must be provided for pointwise calculation."
        assert (
            data.shape[0] == S.shape[0]
        ), "Data rows must match covariance matrix dimension."

    # Variable index sets (past first, future second)
    x = list(range(nx))
    y = list(range(nx, nx + ny))
    a = list(range(nx + ny, nx + ny + nx))
    b = list(range(nx + ny + nx, 2 * (nx + ny)))

    # MI terms (bits)
    Ixa = MI(S, x, a)
    Ixb = MI(S, x, b)
    Iya = MI(S, y, a)
    Iyb = MI(S, y, b)
    Ixya = MI(S, x + y, a)
    Ixyb = MI(S, x + y, b)
    Ixab = MI(S, x, a + b)
    Iyab = MI(S, y, a + b)
    Ixyab = MI(S, x + y, a + b)

    # Broja redundancies via exact_gauss_tilde_pid
    # NB: gpid ordering is [target (dm), source1 (dx), source2 (dy)]
    #     ret_t_sigt=True adds (None, None, None, sig) to the return tuple
    #     res[7] = ri (redundancy, bits)
    #     res[-1] = sig (optimal cross-cov in whitened space)
    def _run_broja(sub_idx, dm, dx, dy):
        """Run exact_gauss_tilde_pid on submatrix, return full result tuple."""
        return exact_gauss_tilde_pid(
            S[np.ix_(sub_idx, sub_idx)],
            dm=dm,
            dx=dx,
            dy=dy,
            ret_t_sigt=pointwise,  # only need sig when pointwise
        )

    # [a+b, x, y] — target=a+b, S1=x, S2=y
    res_xytab = _run_broja(a + b + x + y, dm=nx + ny, dx=nx, dy=ny)
    # [a,   x, y] — target=a,   S1=x, S2=y
    res_xyta = _run_broja(a + x + y, dm=nx, dx=nx, dy=ny)
    # [b,   x, y] — target=b,   S1=x, S2=y
    res_xytb = _run_broja(b + x + y, dm=ny, dx=nx, dy=ny)
    # [x+y, a, b] — target=x+y, S1=a, S2=b
    res_abtxy = _run_broja(x + y + a + b, dm=nx + ny, dx=nx, dy=ny)
    # [x,   a, b] — target=x,   S1=a, S2=b
    res_abtx = _run_broja(x + a + b, dm=nx, dx=nx, dy=ny)
    # [y,   a, b] — target=y,   S1=a, S2=b
    res_abty = _run_broja(y + a + b, dm=ny, dx=nx, dy=ny)

    # Double union (bits)
    if pointwise:
        du_val, covQ_du = double_union_gauss(
            S,
            nx=nx + ny,
            optimiser=optimiser,
            options=options,
            verbose=verbose,
            ret_Q=True,
        )
        double_union_bits = du_val / np.log(2) if not np.isnan(du_val) else np.nan
    else:
        double_union_bits = double_union_gauss(
            S, nx=nx + ny, optimiser=optimiser, options=options, verbose=verbose
        ) / np.log(2)

    if verbose:
        print(f"Broja double union: {double_union_bits:.6f} bits")

    if np.isnan(double_union_bits):
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
        nan_dict = {l: np.nan for l in labels}
        return [nan_dict, nan_dict] if pointwise else nan_dict

    if pointwise:

        # Pointwise MI terms (bits)
        def _pmi(idx1, idx2):
            pt, _ = MI(S, idx1, idx2, pointwise=True, data=data)
            return pt

        pt_Ixa = _pmi(x, a)
        pt_Ixb = _pmi(x, b)
        pt_Iya = _pmi(y, a)
        pt_Iyb = _pmi(y, b)
        pt_Ixya = _pmi(x + y, a)
        pt_Ixyb = _pmi(x + y, b)
        pt_Ixab = _pmi(x, a + b)
        pt_Iyab = _pmi(y, a + b)
        pt_Ixyab = _pmi(x + y, a + b)

        # Pointwise redundancies
        def _pt_red(sub_idx, res_broja, dm, dx, dy):
            """Slice data to match sub_idx ordering and compute pointwise red."""
            covP_sub = S[np.ix_(sub_idx, sub_idx)]
            data_sub = data[sub_idx, :]
            pmi_red, _ = get_pointwise_red(covP_sub, res_broja, dm, dx, dy, data_sub)
            return pmi_red  # already in bits (MI divides by log2 internally)

        pt_Rxytab = _pt_red(a + b + x + y, res_xytab, nx + ny, nx, ny)
        pt_Rxyta = _pt_red(a + x + y, res_xyta, nx, nx, ny)
        pt_Rxytb = _pt_red(b + x + y, res_xytb, ny, nx, ny)
        pt_Rabtxy = _pt_red(x + y + a + b, res_abtxy, nx + ny, nx, ny)
        pt_Rabtx = _pt_red(x + a + b, res_abtx, nx, nx, ny)
        pt_Rabty = _pt_red(y + a + b, res_abty, ny, nx, ny)

        # Pointwise double union
        covP_du = S.copy()
        covQ_du_np = (
            covQ_du if isinstance(covQ_du, np.ndarray) else covQ_du.detach().numpy()
        )
        pt_double_union = get_pointwise_union(
            covP_du, covQ_du_np, nx=nx + ny, ny=nx + ny, data=data, verbose=verbose
        )

    # Build scalar lattice
    scalar_lattice = get_lattice(
        Ixa,
        Ixb,
        Iya,
        Iyb,
        Ixya,
        Ixyb,
        Ixab,
        Iyab,
        Ixyab,
        res_xytab[7],
        res_xyta[7],
        res_xytb[7],
        res_abtxy[7],
        res_abtx[7],
        res_abty[7],
        double_union_bits,
        verbose,
    )

    if not pointwise:
        return scalar_lattice

    # Build pointwise lattice
    pt_lattice = get_lattice(
        pt_Ixa,
        pt_Ixb,
        pt_Iya,
        pt_Iyb,
        pt_Ixya,
        pt_Ixyb,
        pt_Ixab,
        pt_Iyab,
        pt_Ixyab,
        pt_Rxytab,
        pt_Rxyta,
        pt_Rxytb,
        pt_Rabtxy,
        pt_Rabtx,
        pt_Rabty,
        pt_double_union,
        verbose=False,
    )

    return [pt_lattice, scalar_lattice]
