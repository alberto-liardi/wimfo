import sympy as sp
import numpy as np

def get_lattice(Ixa, Ixb, Iya, Iyb, Ixya, Ixyb, Ixab, Iyab, Ixyab, Rxytab, Rxyta, Rxytb, Rabtxy, Rabtx, Rabty, double_union, verbose=False):
    """
    Calculate PhiID lattice and return the PhiID atoms. 
    """
    
    if np.isnan(double_union):
        from warnings import warn
        warn("Warning: Double union is NaN, cannot calculate lattice. Check convergence of double union optimisation.")
        return {"rtr": np.nan, "rta": np.nan, "rtb": np.nan, "rts": np.nan,
                "xtr": np.nan, "xta": np.nan, "xtb": np.nan, "xts": np.nan,
                "ytr": np.nan, "yta": np.nan, "ytb": np.nan, "yts": np.nan,
                "str": np.nan, "sta": np.nan, "stb": np.nan, "sts": np.nan}

    rtr, rta, rtb, rts = sp.var('rtr rta rtb rts')
    xtr, xta, xtb, xts = sp.var('xtr xta xtb xts')
    ytr, yta, ytb, yts = sp.var('ytr yta ytb yts')
    str, sta, stb, sts = sp.var('str sta stb sts')

    ## Set up sympy system of equations
    eqs = [ \
        rtr + rta + xtr + xta - Ixa,
        rtr + rtb + ytr + ytb - Iyb,
        rtr + rtb + xtr + xtb - Ixb,
        rtr + rta + ytr + yta - Iya,
        rtr + rta + xtr + xta + ytr + yta + str + sta - Ixya,
        rtr + rtb + xtr + xtb + ytr + ytb + str + stb - Ixyb,
        rtr + xtr + rta + xta + rtb + xtb + rts + xts - Ixab,
        rtr + ytr + rta + yta + rtb + ytb + rts + yts - Iyab,
        rtr + xtr + ytr + str + rta + xta + yta + sta + rtb + xtb + ytb + stb + rts + xts + yts + sts - Ixyab,
        rtr + rta + rtb + rts - Rxytab,
        rtr + rta - Rxyta,
        rtr + rtb - Rxytb,
        rtr + xtr + ytr + str - Rabtxy,
        rtr + xtr - Rabtx,
        rtr + ytr - Rabty,
        str + sta + stb + sts + rts + xts + yts - (Ixyab - double_union)
        ]

    ## Solve and print
    all_pid = [rtr, rta, rtb, rts, \
                xtr, xta, xtb, xts, \
                ytr, yta, ytb, yts, \
                str, sta, stb, sts]

    m, b = sp.linear_eq_to_matrix(eqs, all_pid)
    v = np.linalg.lstsq(np.matrix(m, dtype=float), np.matrix(b, dtype=float), rcond=None)[0]

    v = [0 if abs(val) < 1e-7 * double_union else val.item() for val in v]

    res = {}
    for n,a in zip(all_pid, np.round(v, 5)):
        res[n.name] = a
        if verbose:
            print('%s: %f' % (n, a))

    return res