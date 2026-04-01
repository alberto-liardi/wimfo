import torch
import numpy as np


def mutual_information(Q, n):
    # Mutual information in BITS

    if len(Q.shape) == 1:
        # Reshape P into a 4D tensor
        Q = Q.reshape((n, n, n, n))

    P_X = marginalise(Q, (2, 3))
    P_Y = marginalise(Q, (0, 1))

    H_X = -torch.sum(P_X * torch.log2(P_X + 1e-10))
    H_Y = -torch.sum(P_Y * torch.log2(P_Y + 1e-10))
    H_XY = -torch.sum(Q * torch.log2(Q + 1e-10))  # Small epsilon to avoid log2(0)

    return H_X + H_Y - H_XY


def check_constraints(Q, P, n, atol=1e-4, rtol=1e-4):
    for i in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        assert torch.allclose(
            marginalise(P.reshape((n, n, n, n)), i),
            marginalise(Q.reshape((n, n, n, n)), i),
            atol=atol,
            rtol=rtol,
        ), f"Constraint {i} not satisfied!"


# Define the marginalize function
def marginalise(P, indx):
    return P.sum(dim=indx)


# Define the projection function
def project(Q, P, max_iterations=1000, tolerance=1e-6, eps=1e-30):
    # Compute target marginals from P
    A = marginalise(P, (0, 2))
    B = marginalise(P, (0, 3))
    C = marginalise(P, (1, 2))
    D = marginalise(P, (1, 3))

    # Iterative proportional fitting
    for _ in range(max_iterations):
        q_old = Q.clone()

        # Enforce marginal constraints
        q_margin = Q.sum(dim=(0, 2), keepdim=True)
        Q = Q * (A[None, :, None, :] / torch.clamp(q_margin, min=eps))

        q_margin = Q.sum(dim=(0, 3), keepdim=True)
        Q = Q * (B[None, :, :, None] / torch.clamp(q_margin, min=eps))

        q_margin = Q.sum(dim=(1, 2), keepdim=True)
        Q = Q * (C[:, None, None, :] / torch.clamp(q_margin, min=eps))

        q_margin = Q.sum(dim=(1, 3), keepdim=True)
        Q = Q * (D[:, None, :, None] / torch.clamp(q_margin, min=eps))

        # Check for convergence
        if torch.max(torch.abs(Q - q_old)) < tolerance:
            break

    return Q


def double_union_discrete(
    P, Q=None, n=2, verbose=False, optimiser="Mirror", options=None, **kwargs
):

    # TODO: possible improvements for Mirror optimiser:
    # Replace autograd with a closed-form MI gradient: e.g. grad = log Q - log PX - log PY
    # Project less often:  project every N mirror steps, then do a few extra projection iterations at the end.

    if optimiser == "Newton":
        optimiser = "newton-exact"

    if not isinstance(P, torch.Tensor):
        P = torch.tensor(P, dtype=torch.float64)

    # Initialize joint distribution Q as a random tensor
    if Q is None:
        Q = torch.rand(n, n, n, n, requires_grad=True)
    else:
        Q = torch.tensor(Q, dtype=torch.float64, requires_grad=True)

    try:
        P_tensor = P.reshape((n, n, n, n))
        Q = Q.reshape((n, n, n, n))
    except:
        raise ValueError(
            f"Wrong input distribution size, \
                         it needs to be reshaped into a {n}x{n}x{n}x{n} tensor."
        )

    # avoid numerical instabilities due to zero probabilities
    P = P + 1e-10
    Q = Q + 1e-10

    if options is None:
        options = {}

    # Mirror-descent branch
    if optimiser == "Mirror":
        if verbose:
            print("Optimising with Mirror...")

        steps = int(options.get("steps", 200))
        lr = float(options.get("lr", 1e-1))
        proj_max_iters = int(options.get("proj_max_iters", 500))
        atol = float(options.get("atol", 1e-4))
        rtol = float(options.get("rtol", 1e-4))
        window_size = int(options.get("window_size", 10))

        # Ensure Q is normalized and positive, then set requires_grad
        with torch.no_grad():
            Q = Q / torch.sum(Q)
            Q = torch.clamp(Q, min=1e-30)
        Q.requires_grad_(True)

        MIs = []
        for t in range(steps):
            # Normalize inside the loop for MI
            Q_norm = Q / torch.sum(Q)
            MI = mutual_information(Q_norm, n)

            MIs.append(MI.clone().detach())
            if t > window_size and torch.allclose(
                MI, torch.tensor(MIs[-window_size:-1]), atol=atol, rtol=rtol
            ):
                # if check_constraints(Q, P_tensor, n, atol=atol, rtol=rtol):
                break
            
            # if verbose and (t % 10 == 0):
            #     print(f"Step {t}: MI={MI.item():.6f}")

            # Compute gradient directly to avoid second-backward issues
            grad = torch.autograd.grad(
                MI, Q, retain_graph=False, create_graph=False, allow_unused=False
            )[0]

            # Mirror-descent update (multiplicative/exponentiated)
            with torch.no_grad():
                Q = Q * torch.exp(-lr * grad)
                Q = torch.clamp(Q, min=1e-30)
                Q = Q / torch.sum(Q)

            Q.requires_grad_(True)
            # Project back to constraints WITHOUT tracking gradients
            with torch.no_grad():
                Q = project(Q, P_tensor, max_iterations=proj_max_iters)
            Q.requires_grad_(True)

        # Final projection and evaluation
        with torch.no_grad():
            Q_final = project(Q, P_tensor, max_iterations=proj_max_iters)
        check_constraints(Q_final, P_tensor, n, **kwargs)
        MI_final = mutual_information(Q_final, n).item()
        return float(MI_final)

    elif optimiser == "newton-exact" or optimiser == "newton-cg":
        from torchmin import minimize

        options.update({"custom_wolfe": True})

        def loss(Q):
            # Project Q back onto the feasible set defined by the constraints
            Q_renom = Q / torch.sum(Q)
            Q_proj = project(Q_renom, P_tensor)
            MI = mutual_information(Q_proj, n)
            # if verbose: print(MI)
            return MI

        if verbose:
            print(f"Optimising with {optimiser}...")
        result = minimize(loss, Q, method=optimiser, options=options)  # 'newton-cg'

        x = result.x
        if not result.success:
            print(f"CONVERGENCE NOT REACHED! Trying again...")
            return np.nan

        # check the constraints are satisfied
        check_constraints(P_tensor, project(x / torch.sum(x), P_tensor), n, **kwargs)

        return float(loss(x).detach())

    else:
        raise ValueError(
            f"Optimiser {optimiser} not recognised. Use 'Mirror', 'newton-exact' or 'newton-cg'."
        )
