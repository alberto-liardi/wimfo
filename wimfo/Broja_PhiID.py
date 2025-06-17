import numpy as np
from wimfo.utils.utils_gauss import get_cov
from wimfo.gaussian.broja_phiid_gauss import broja_phiid
from wimfo.discrete.broja_phiid_discrete import broja_phiid_discrete
from wimfo.utils.utils_discrete import estimate_discrete_distribution


def PhiID(
    input,
    t=1,
    option="data",
    type="gaussian",
    unit="bits",
    verbose=False,
    optimiser="Adam",
    options={"atol": 1e-4, "rtol": 1e-4},
    **kwargs,
):
    """
    Compute Broja-PhiID from data or probability distribution.

    Parameters:
    - input (numpy.ndarray):          Data matrix (variables × samples), covariance matrix (if type == "gaussian"),
                                      or 2×2×2×2 probability distribution (if type == "discrete").
    - t (int):                        Future lag step.
    - option (str):                   Either "data" or "distr". For "gaussian", this determines whether a data matrix
                                      or covariance matrix is passed. For "discrete", use a probability distribution or binary data.
    - unit (str, optional):           Unit of information. Either "bits" or "nats". Default is "bits".
    - verbose (bool, optional):       If True, prints additional information during computation. Default is False.
    - optimiser (str, optional):      Optimiser to use. Options are "Adam" or "Newton". For large systems (>15 variables), use "Adam".
    - options (dict, optional):       Dictionary of options for the optimiser. Default is None.

    Returns:
    - dict:                           Broja PhiID atoms (in bits or nats).
    """


    if unit not in ["bits", "nats"]:
        raise ValueError(f"Unit must be either 'bits' or 'nats', {unit} was passed.")

    # gaussian data
    if type == "gaussian":
        if option == "data" and input.shape[0] == input.shape[1]:
            print("Warning: data is square, perhaps you meant to use option='cov'")
        if option == "data":
            input = get_cov(input, t=t)

        phiid_dict = broja_phiid(
            input, verbose=verbose, optimiser=optimiser, options=options, **kwargs
        )

    # discrete data
    elif type == "discrete":
        if option == "data":
            assert (
                input.shape[0] == 4
            ), "Discrete data must have 4 rows (source 1, source 2, target 1, target 2)."
            if not np.all(np.isin(input, [0, 1])):
                print(
                    "Warning: Discrete data should be binary (0 or 1). Converting to binary."
                )
                input = (input > input.mean(input, axis=1, keepdims=True)).astype(int)
            input = estimate_discrete_distribution(*input)

        phiid_dict = broja_phiid_discrete(input, verbose=False, **kwargs)

    else:
        raise ValueError(
            f"Type must be either 'gaussian' or 'discrete', {type} was passed."
        )

    # convert dimensions if needed
    if unit == "nats":
        for key in phiid_dict:
            phiid_dict[key] *= np.log(2)

    return phiid_dict
