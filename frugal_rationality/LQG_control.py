import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.linalg import inv
from scipy.linalg import solve_discrete_lyapunov

# ==================================================================================
# This function solves the problem when the inference penalty is set to zero
# Args:
#  (A, B, H, Q, R), The parameters of the linearized world model
#  (Cx, Cu), The parameters of the loss function, Cn is assumed to be zero
# Output:
#   K, the optimal weight new observations deserve when calculating state estimates
#   L, the optimal control gain that casts estimates into actions
#   Sigma, the joint state and action covariance matrix that K and L generate
#   State and action costs, the expected performance given K and L
# ==================================================================================

def solve(A, B, H, Q, R, Cx, Cu):

    dim = A.shape[0]    # dimensionality of state

    # Control via a linear quadratic regulator
    Pc = solve_discrete_are(A, B, Cx, Cu)
    # control gain
    L = -inv(Cu + B.T @ Pc @ B) @ B.T @ Pc @ A

    # Inference via a Kalman filter
    tmpA = A.T
    tmpB = A.T @ H.T
    tmpQ = Q
    tmpR = H @ Q @ H.T + R
    tmpE = np.eye(dim)
    tmpS = Q @ H.T

    Pf = solve_discrete_are(tmpA, tmpB, tmpQ, tmpR, tmpE, tmpS)
    # Optimal weight to new observations (Kalman gain)
    K = Pf @ H.T @ inv(R)
    # Optimal weight to prior (after propagated 1 time step)
    G = np.dot(np.eye(A.shape[0]) - np.dot(K, H), A + np.dot(B, L))

    # Compute expected performance (state and action costs)
    # M := Transition matrix of augmented state z = [s, mu]
    M = np.block([[A, B @ L], [K @ H @ A, (np.eye(dim) - K @ H) @ (A + B @ L) + K @ H @ B @ L]])

    # augQ := Process noise in joint state and state_estimate space
    augQ = np.block([[Q, (K @ H @ Q).T], [K @ H @ Q, K @ H @ Q @ H.T @ K.T + K @ R @ K.T]])

    # Joint covariance (states and state_estimates) at equilibrium
    Sigma_mu = solve_discrete_lyapunov(M, augQ)

    # Joint covariance (states and actions)  at equilibrium
    Sigma = np.block([[Sigma_mu[:dim, :dim], Sigma_mu[:dim, dim:] @ L.T],
                      [L @ Sigma_mu[dim:, :dim], L @ Sigma_mu[dim:, dim:] @ L.T]])

    state_cost = np.trace(Cx @ Sigma[:dim, :dim])
    action_cost = np.trace(Cu @ Sigma[dim:, dim:])

    det_X = np.linalg.det(Sigma[:dim, :dim])
    det_U = np.linalg.det(Sigma[dim:, dim:])
    det_S = np.linalg.det(Sigma)
    info_cost = 0.5 * (np.log2(det_X * det_U / det_S))

    res = {'L': L, 'K': K, 'G': G, 'S': Sigma,
            'state_cost': state_cost, 'action_cost': action_cost, 'bits': info_cost}

    return res