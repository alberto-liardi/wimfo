import numpy as np

from dit import Distribution
from dit.pid import PID_BROJA

from wimfo.utils.phiid_lattice import get_lattice
from wimfo.discrete.double_union_discrete import double_union_discrete


def get_PID_MI(pid):

    # NB: redundancy, unique0, unique1, synergy
    p = np.zeros(4)
    m = np.zeros(4)
    for i, n in enumerate([((0,), (1,)), ((0,),), ((1,),), ((0, 1),)]):
        # print(pid._pis)
        p[i] = pid._pis[n]
        m[i] = pid._reds[n]

    return p, m[1:]


def broja_phiid_discrete(probs, verbose=False, **kwargs):

    probs+=1e-12 # alleviate numerical instabilities

    out_joint_PID = [
        "00A",
        "00B",
        "00C",
        "00D",
        "01A",
        "01B",
        "01C",
        "01D",
        "10A",
        "10B",
        "10C",
        "10D",
        "11A",
        "11B",
        "11C",
        "11D",
    ]

    out_PID = ["000", "001", "010", "011", "100", "101", "110", "111"]

    # xy2ab
    pid_xytab, mis = get_PID_MI(PID_BROJA(Distribution(out_joint_PID, probs.flatten())))
    Rxytab = pid_xytab[0]
    Ixab = mis[0]
    Iyab = mis[1]
    Ixyab = mis[2]
    # xy2a
    pid_xyta, mis = get_PID_MI(
        PID_BROJA(Distribution(out_PID, probs.sum(axis=3).flatten()))
    )
    Rxyta = pid_xyta[0]
    Ixa = mis[0]
    Iya = mis[1]
    Ixya = mis[2]
    # xy2b
    pid_xytb, mis = get_PID_MI(
        PID_BROJA(Distribution(out_PID, probs.sum(axis=2).flatten()))
    )
    Rxytb = pid_xytb[0]
    Ixb = mis[0]
    Iyb = mis[1]
    Ixyb = mis[2]
    # ab2xy
    pid_abtxy, _ = get_PID_MI(
        PID_BROJA(Distribution(out_joint_PID, probs.transpose(2, 3, 0, 1).flatten()))
    )
    Rabtxy = pid_abtxy[0]
    # abtx
    pid_abtx, _ = get_PID_MI(
        PID_BROJA(
            Distribution(out_PID, probs.transpose(2, 3, 0, 1).sum(axis=3).flatten())
        )
    )
    Rabtx = pid_abtx[0]
    # abty
    pid_abty, _ = get_PID_MI(
        PID_BROJA(
            Distribution(out_PID, probs.transpose(2, 3, 0, 1).sum(axis=2).flatten())
        )
    )
    Rabty = pid_abty[0]

    ## Calculate PhiID union information
    double_union = np.nan
    limit = 10
    while np.isnan(double_union) and limit > 0:
        if limit == 10:
            p0 = probs
        else:
            p0 = None
        double_union = double_union_discrete(probs, p0, verbose, **kwargs)
        limit-=1
    if verbose:
        print(f"Broja double union is {double_union}")

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
        verbose,
    )
