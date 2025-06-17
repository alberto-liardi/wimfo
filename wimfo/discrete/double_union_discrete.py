import torch
from torchmin import minimize
import numpy as np


def mutual_information(Q):
    # Mutual information in BITS

    if len(Q.shape) == 1:
        # Reshape P into a 4D tensor
        Q = Q.reshape((2, 2, 2, 2))

    P_X = marginalise(Q, (2, 3))
    P_Y = marginalise(Q, (0, 1))

    H_X = -torch.sum(P_X * torch.log2(P_X + 1e-10))
    H_Y = -torch.sum(P_Y * torch.log2(P_Y + 1e-10))
    H_XY = -torch.sum(Q * torch.log2(Q + 1e-10))  # Small epsilon to avoid log2(0)

    return H_X + H_Y - H_XY


def check_constraints(Q, P, atol=1e-4, rtol=1e-4):
    for i in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        assert torch.allclose(
            marginalise(P.reshape((2, 2, 2, 2)), i),
            marginalise(Q.reshape((2, 2, 2, 2)), i),
            atol=atol,
            rtol=rtol,
        ), f"Constraint {i} not satisfied!"


# Define the marginalize function
def marginalise(P, indx):
    return P.sum(dim=indx)


# Define the projection function
def project(Q, P):
    # Compute target marginals from P
    A = marginalise(P, (0, 2))
    B = marginalise(P, (0, 3))
    C = marginalise(P, (1, 2))
    D = marginalise(P, (1, 3))

    # Iterative proportional fitting
    tolerance = 1e-6
    max_iterations = 1000

    for _ in range(max_iterations):
        q_old = Q.clone()

        # Enforce marginal constraints
        q_margin = Q.sum(dim=(0, 2), keepdims=True)
        Q = Q * A[None, :, None, :] / q_margin

        q_margin = Q.sum(dim=(0, 3), keepdims=True)
        Q = Q * B[None, :, :, None] / q_margin

        q_margin = Q.sum(dim=(1, 2), keepdims=True)
        Q = Q * C[:, None, None, :] / q_margin

        q_margin = Q.sum(dim=(1, 3), keepdims=True)
        Q = Q * D[:, None, :, None] / q_margin

        # Check for convergence
        if torch.max(torch.abs(Q - q_old)) < tolerance:
            break

    return Q


def double_union_discrete(P, verbose=False, **kwargs):

    if not isinstance(P, torch.Tensor):
        P = torch.tensor(P, dtype=torch.float64)

    # Initialize joint distribution Q as a random tensor
    Q = torch.rand(2, 2, 2, 2, requires_grad=True)
    P_tensor = P.reshape((2, 2, 2, 2))

    def loss(Q):
        # Project Q back onto the feasible set defined by the constraints
        Q_renom = Q / torch.sum(Q)
        Q_proj = project(Q_renom, P_tensor)
        MI = mutual_information(Q_proj)
        # if verbose: print(MI)
        return MI

    result = minimize(loss, Q, method="newton-exact")  # 'newton-cg'

    x = result.x
    if not result.success:
        print(f"CONVERGENCE NOT REACHED! Trying again...")
        return np.nan

    # check the constraints are satisfied
    check_constraints(P_tensor, project(x / torch.sum(x), P_tensor), **kwargs)

    return float(loss(x).detach())
