import numpy as np


def get_lattice(
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
    verbose=False,
):
    """
    Calculate PhiID lattice atoms from Broja redundancies and MI terms.
    Works for both scalar (non-pointwise) and array (pointwise) inputs.
    All inputs must be in BITS.

    Returns:
        dict of scalars or arrays with keys:
        rtr, rta, rtb, rts, xtr, xta, xtb, xts,
        ytr, yta, ytb, yts, str, sta, stb, sts
    """
    # Check for NaN double_union (scalar case only)
    if np.isscalar(double_union) and np.isnan(double_union):
        from warnings import warn

        warn("Double union is NaN, cannot calculate lattice.")
        nan = np.nan
        return {
            n: nan
            for n in [
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
        }

    names = [
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

    # Coefficient matrix A (16x16)
    # Equations :
    # 1)  rtr + rta + xtr + xta                                    = Ixa
    # 2)  rtr + rtb + ytr + ytb                                    = Iyb
    # 3)  rtr + rtb + xtr + xtb                                    = Ixb
    # 4)  rtr + rta + ytr + yta                                    = Iya
    # 5)  rtr + rta + xtr + xta + ytr + yta + str + sta            = Ixya
    # 6)  rtr + rtb + xtr + xtb + ytr + ytb + str + stb            = Ixyb
    # 7)  rtr + xtr + rta + xta + rtb + xtb + rts + xts            = Ixab
    # 8)  rtr + ytr + rta + yta + rtb + ytb + rts + yts            = Iyab
    # 9)  all atoms                                                 = Ixyab
    # 10) rtr + rta + rtb + rts                                    = Rxytab
    # 11) rtr + rta                                                 = Rxyta
    # 12) rtr + rtb                                                 = Rxytb
    # 13) rtr + xtr + ytr + str                                    = Rabtxy
    # 14) rtr + xtr                                                 = Rabtx
    # 15) rtr + ytr                                                 = Rabty
    # 16) str + sta + stb + sts + rts + xts + yts                  = Ixyab - double_union
    A = np.zeros((16, 16))
    A[0, [0, 1, 4, 5]] = 1  # Ixa
    A[1, [0, 2, 8, 10]] = 1  # Iyb
    A[2, [0, 2, 4, 6]] = 1  # Ixb
    A[3, [0, 1, 8, 9]] = 1  # Iya
    A[4, [0, 1, 4, 5, 8, 9, 12, 13]] = 1  # Ixya
    A[5, [0, 2, 4, 6, 8, 10, 12, 14]] = 1  # Ixyb
    A[6, [0, 4, 1, 5, 2, 6, 3, 7]] = 1  # Ixab
    A[7, [0, 8, 1, 9, 2, 10, 3, 11]] = 1  # Iyab
    A[8, :] = 1  # Ixyab
    A[9, [0, 1, 2, 3]] = 1  # Rxytab
    A[10, [0, 1]] = 1  # Rxyta
    A[11, [0, 2]] = 1  # Rxytb
    A[12, [0, 4, 8, 12]] = 1  # Rabtxy
    A[13, [0, 4]] = 1  # Rabtx
    A[14, [0, 8]] = 1  # Rabty
    A[15, [3, 7, 11, 12, 13, 14, 15]] = 1  # Ixyab - double_union

    A_inv = np.linalg.inv(A)

    # Stack RHS — works for both scalars and arrays
    rhs_vals = [
        Ixa,
        Iyb,
        Ixb,
        Iya,
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
        Ixyab - double_union,
    ]

    B = np.array(rhs_vals, dtype=float)  # (16,) or (16,N)
    X = A_inv @ B  # (16,) or (16,N)

    res = {}
    for i, name in enumerate(names):
        val = X[i]
        # Zero out near-zero values (relative to double_union magnitude)
        tol = 1e-7 * (
            np.abs(double_union)
            if np.isscalar(double_union)
            else np.abs(np.mean(double_union))
        )
        if np.isscalar(val):
            res[name] = 0.0 if abs(val) < tol else float(np.round(val, 5))
        else:
            res[name] = val  # keep full array for pointwise
        if verbose:
            mean_val = val if np.isscalar(val) else np.mean(val)
            print(f"{name}: {mean_val:.6f}")

    return res
