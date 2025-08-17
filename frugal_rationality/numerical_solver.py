import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import sqrtm
from numpy.linalg import inv
from scipy.optimize import approx_fprime
from numdifftools import Hessian

# ===================================================================================================
# Numerical solution to a meta-cognitive POMDP with linear-Gaussian dynamics
# Args:
#   (A, B, H, Q, R), world model parameters
#   (Cx, Cu, cn), loss function parameters
#    ini, initial guess
#    learning_rate of optimizer
#    max_steps, max number of iterations during optimization
#    hess_step, quality of Hessian
# Output:
#   frugal_strategy, parameters of the frugal strategy (or the candidate after max_step iterations)
#   min_value, minimum value of the loss function (or the value after max_step iterations)
#   success, flag indicating min_value is a local minima (True)
# ===================================================================================================
def minimize(A, B, H, Q, R, Cx, Cu, cn, ini, learning_rate=0.001, max_steps=200000, hess_step=1e-6):

    # Cast model params and loss function params
    A_tf = tf.Variable(A.tolist())
    B_tf = tf.Variable(B.tolist())
    H_tf = tf.Variable(H.tolist())
    Q_tf = tf.Variable(Q.tolist())
    R_tf = tf.Variable(R.tolist())
    Cx_tf = tf.Variable(Cx.tolist())
    Cu_tf = tf.Variable(Cu.tolist())
    args = [A_tf, B_tf, H_tf, Q_tf, R_tf, Cx_tf, Cu_tf, cn]

    # Initial guess (usually unconstrained solution calculated using LQG)
    Psi = tf.Variable(ini[0].tolist())
    Pi = tf.Variable(ini[1].tolist())
    flattened_params = tf.Variable(tf.concat([tf.reshape(Psi, [-1]), tf.reshape(Pi, [-1])], axis=0))

    # Initialize Adam optimizer
    optimizer = Adam(learning_rate=learning_rate,  # Initial learning rate
                                beta_1=0.9,                   # Decay rate for the first moment
                                beta_2=0.999,                 # Decay rate for the second moment
                                epsilon=1e-7                  # Small constant for numerical stability)
                                                )
    # Training loop
    success = False
    for step in range(max_steps):

        # Calculate the gradient of loss function with respect to the parameters of the cantidate strategy (Pi, Psi)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([flattened_params])
            loss = objectiveFunction(flattened_params, args)
        gradient = tape.gradient(loss, [flattened_params])

        # Check progress every 100 steps
        if step % 100 == 0:
            print(f'Step {step}: loss = {loss.numpy()}')

            # Calculate Hessian at critical point to check curvature
            hessian = fun_hess(flattened_params.numpy(), [A, B, H, Q, R, Cx, Cu, cn, hess_step]).astype(np.float32)
            eigs = np.linalg.eigvals(hessian)
            det = np.linalg.det(hessian)

            # If positive eigenvalues and determinant, we got a local min!!
            if np.all(eigs > 0) and det > 0:
                print("Success")
                print(f'Minimum value of the function: {loss.numpy()}')
                success = True
                break
            else:
                # print eigenvalues that remain less than zero
                print("working on: ", eigs[eigs < 0])

            # Use information from the Hessian to guide the learning rate
            gradient = tf.linalg.matvec(inv(hessian), gradient)

        # Apply gradients
        optimizer.apply_gradients(zip(gradient, [flattened_params]))

    frugal_strategy = flattened_params.numpy()
    min_value = loss.numpy()

    if not success:
        print(f'No local minimum, value of the function after {max_steps} iterations: {loss.numpy()}')

    return frugal_strategy, min_value, success

# ===================================================================================================
# Differentiable loss function
# Args:
#   flattened_params, TF calculates gradients with respect to this params
#   args, world model and loss function params
# Output:
#   value of the loss evaluated at flattened_params
# ===================================================================================================
def objectiveFunction(flattened_params, args):
    A, B, H, Q, R, Cx, Cu, cn = args       # Model parameters

    dim_s = A.shape[0]  # Shape of state
    dim_u = B.shape[1]  # Shape of action
    dim_y = H.shape[0]  # Shape of observation

    # Rebuilt parameters of candidate strategy (both are matrices)
    Psi = tf.reshape(flattened_params[:dim_u * dim_y], [dim_u, dim_y])
    Pi = tf.reshape(flattened_params[dim_u * dim_y:], [dim_u, dim_u])

    # Create augmented transition matrix M describing the joint evolution of states and actions
    top_row = tf.concat([A, B], axis=1)
    bottom_row = tf.concat([Psi @ H @ A, Pi + Psi @ H @ B], axis=1)
    M = tf.concat([top_row, bottom_row], axis=0)

    # Create matrix P describing the stochasticity in the joint space of states and actions
    top_row = tf.concat([Q, tf.transpose(Psi @ H @ Q)], axis=1)
    bottom_row = tf.concat([Psi @ H @ Q, Psi @ H @ Q @ tf.transpose(H) @ tf.transpose(Psi) + Psi @ R @ tf.transpose(Psi)], axis=1)
    P = tf.concat([top_row, bottom_row], axis=0)

    # Calculate joint covariance matrix describing state and actions at equilibrium
    # Sigma is the solution to the discrete Lyapunov equation S = M S M.T + P
    dim = tf.shape(M)[0]
    Q_flat = tf.reshape(P, [-1])
    I_dim = tf.eye(dim * dim, dtype=tf.float32)
    M_kron = tf.linalg.LinearOperatorKronecker(
        [tf.linalg.LinearOperatorFullMatrix(M), tf.linalg.LinearOperatorFullMatrix(M)]).to_dense()
    tmp = I_dim - M_kron
    inv_tmp = tf.linalg.inv(tmp)
    sol_flat = tf.linalg.matvec(inv_tmp, Q_flat)
    Sigma = tf.reshape(sol_flat, [dim, dim])

    # Calculate mutual information between states and estimates
    det_X = tf.linalg.det(Sigma[: dim_s, : dim_s])
    det_U = tf.linalg.det(Sigma[dim_s:, dim_s:])
    det_S = tf.linalg.det(Sigma)
    bits = 0.5 * (tf.math.log(det_X * det_U / det_S) / tf.math.log(2.))

    # Calculate action and state costs
    task_cost = tf.linalg.trace(Cx @ Sigma[:dim_s, :dim_s]) + tf.linalg.trace(Cu @ Sigma[dim_s:, dim_s:])

    # Soft constraints ensuring controllability and plausibility of trajectory distributions
    eigenvalues_M = tf.math.real(tf.linalg.eigvals(M))
    eigenvalues_S = tf.math.real(tf.linalg.eigvals(Sigma))
    is_valid = (tf.reduce_all(eigenvalues_M > -1) and
                tf.reduce_all(eigenvalues_M < 1) and
                tf.reduce_all(eigenvalues_S > 0) and
                det_X > 0 and
                det_U > 0 and
                det_S > 0)

    # Return loss if the candidate strategy generated a valid solution
    if is_valid:
        return task_cost + cn * bits

    # Otherwise, return a heavy penalty
    return 1e10 + task_cost ** 2

# ===================================================================================================
# Calculate quickly the Hessian to inform the optimizer´s learning rate and the quality of solution
# ===================================================================================================
def fun_hess(params, args):
    A, B, H, Q, R, Cx, Cu, cn, my_step = args
    return Hessian(lambda params: getExpectedPerformance(params, A, B, H, Q, R, Cx, Cu, cn), step=my_step)(params)

# ===================================================================================================
# Quick evaluation of loss function.
# This function uses block matrices and scipy.linalg.solve_discrete_lyapunov() to quickly
# evaluate the loss given the params of the frugal strategy
# WARNING: TF cannot calculate gradients using this loss function!
# Args:
#   params, flattened parameters of the frugal strategy
#   (A, B, H, Q, R), world model parameters
#   (Cx, Cu, cn), loss function parameters
#   just_loss, if True return the total loss, otherwise return everything
# Output:
#   Psi, Controller´s attention
#   Pi, Controller´s base dynamics
#   M, transition matrix describing joint evolution of states and actions
#   Sigma, Joint covariance of states and actions
#   individual costs: info, state, action
# ===================================================================================================
def getExpectedPerformance(params, A, B, H, Q, R, Cx, Cu, cn, just_loss=True):

    dim_s = A.shape[0]  # Shape of state
    dim_u = B.shape[1]  # Shape of action
    dim_y = H.shape[0]  # Shape of observation

    # Rebuilt parameters of candidate strategy (both are matrices)
    Psi = np.array(params[: dim_u * dim_y]).reshape([dim_u, dim_y])
    Pi = np.array(params[dim_u * dim_y:]).reshape([dim_u, dim_u])

    # Create augmented transition matrix M describing the joint evolution of states and actions
    M = np.block([[A, B], [Psi @ H @ A, Pi + Psi @ H @ B]])
    # Create matrix P describing the stochasticity in the joint space of states and actions
    P = np.block([[Q, (Psi @ H @ Q).T], [Psi @ H @ Q, Psi @ H @ Q @ H.T @ Psi.T + Psi @ R @ Psi.T]])

    # Calculate joint covariance matrix describing state and actions at equilibrium
    # Sigma is the solution to the discrete Lyapunov equation S = M S M.T + P
    Sigma = solve_discrete_lyapunov(M, P) # Covariance over x and u

    # Calculate mutual information between states and estimates
    det_X = np.linalg.det(Sigma[:dim_s, :dim_s])
    det_U = np.linalg.det(Sigma[dim_s:, dim_s:])
    det_S = np.linalg.det(Sigma)
    bits = 0.5 * (np.log2(det_X * det_U / det_S))

    # Calculate action and state costs
    state_cost = np.trace(Cx @ Sigma[:dim_s, :dim_s])
    action_cost = np.trace(Cu @ Sigma[dim_s:, dim_s:])

    if just_loss:
        return state_cost + action_cost + cn * bits

    return Psi, Pi, M, Sigma, bits, state_cost, action_cost

# ===================================================================================================
# Recovery of statistics generated by a frugal strategy
# Args:
#   (A, B, H, Q, R), world model parameters
#   (Cx, Cu, cn), loss function parameters
#   params, flattened parameters of the frugal strategy
# Output:
#   res, dictionary (see the content below)
# ===================================================================================================
def readSol(A, B, H, Q, R, Cx, Cu, cn, params):

    dim_s = A.shape[0]  # Shape of state

    # Get expected performance of frugal strategy
    Psi, Pi, M, Sigma, bits, state_cost, action_cost = getExpectedPerformance(params, A, B, H, Q, R, Cx, Cu, cn, just_loss=False)

    # Calculate the parameters of the inference (K, G) and controller (L)
    # This can be computed only if H and B are squared matrices
    if H.shape[0] == H.shape[1] and B.shape[0] == B.shape[1]:
        K = Sigma[dim_s:, :dim_s] @ inv(Sigma[dim_s:, dim_s:]) @ Psi
        L = Psi @ inv(K)
        G = inv(L) @ Pi @ inv(L).T
    else:
        K, L, G = None, None, None

    res = {'M': M,  'S': Sigma,                # Transition matrix and joint covariance
           'L': L, 'K': K, 'G': G,             # Inference and control params
           'Psi': Psi, 'Pi': Pi,               # Params of controller in input-output form
           'bits': bits,                       # Information gained via inference
           'state': state_cost,                # Task performance
           'action': action_cost,              # Motion effort
           'info_cost': cn * bits              # Inference cost
           }

    return res

# ===================================================================================================
# Recover complete family from single solution
# Args:
#   (A, B, H, Q, R), world model parameters
#   (Cx, Cu, cn), loss function parameters
#   res, single solution
#   members, family param is continuous, how many discrete elements you want to recover?
# Output:
#   family, a list with [members] strategies, everyone achieving the same statistical performance
# ===================================================================================================
def getFamily(A, B, H, Q, R, Cx, Cu, cn, res, members=2):

    dim = A.shape[0]
    S = res['S']

    iSx = inv(S[:dim, :dim])
    Su = S[dim:, dim:]
    Sxu = S[dim:, :dim]
    Sy_ = inv(H) @ R @ inv(H).T - S[:dim, :dim]
    Sdin = Sxu @ A.T + Su @ B.T

    # Ellipsoidal constraint that all family members meet is a function of F0, F1, F2
    F0 = Su + Sdin @ iSx @ Sy_ @ iSx.T @ Sdin.T
    F1 = Sdin @ iSx @ Sxu.T + Sdin @ iSx @ Sy_ @ iSx.T @ Sxu.T
    F3 = Su - 2 * (Sxu @ iSx @ Sxu.T) - Sxu @ iSx @ Sy_ @ iSx.T @ Sxu.T

    # Eigendecompose remaining uncertainty
    L1, D1 = np.linalg.eigh(F0)
    L2, D2 = np.linalg.eigh(F3 + F1.T @ inv(F0).T @ F1)

    # Free rotation, dimensionality depends on dimensionality of actions
    if B.shape[1] == 1:
        rotations = [np.array([[1]]), np.array([[-1]])]
    else:
        rad_ops = np.linspace(0, 2*np.pi, members)
        rotations = [np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]) for rad in rad_ops]

    # Generate complete family
    family = []
    for i, Th in enumerate(rotations):

        new_Pi = D2 @ sqrtm(np.diag(L2)) @ Th @ inv(sqrtm(np.diag(L1))) @ inv(D1) + F1.T @ inv(F0)
        new_Psi = Sxu @ iSx @ inv(H) - new_Pi @ Sdin @ iSx @ inv(H)
        new_rat_params = np.concatenate([new_Psi.flatten(), new_Pi.flatten()])

        # Get expected performance of new solution
        new_rat_res = readSol(A, B, H, Q, R, Cx, Cu, cn, new_rat_params)

        # Make sure solution yields the original covariance matrix over the states and actions
        if np.allclose(S, new_rat_res['S']):
            family.append(new_rat_res)
        else:
            print(f"ERROR: Family member {i} do NOT yield the same covariance matrix")
            return []

    return family

# ===================================================================================================
# Next 2 functions calculate the natural gradient of loss with respect to changes in model params
# Args:
#   params, [mass, arm length]
#   args = [A, B, H, Q, R, Cx, Cu, cn, dt, strategy]
# Output:
#   loss(m, l)
# ===================================================================================================
# loss as a function of m and l
def generalizationFunction(params, args):
    A, _, H, Q, R, Cx, Cu, cn, dt, strategy = args

    m, l = params
    B = np.array([[0., 0.], [0., 0.], [0., 0.], [dt / m, dt / m], [0., 0.],
                      [dt * l / (2 * m * l ** 2), - dt * l / (2 * m * l ** 2)]])

    return getExpectedPerformance(strategy, A, B, H, Q, R, Cx, Cu, cn, just_loss=True)

# Calculate natural gradient to capture worst-case local sensitivity
def getGeneralization(A, B, H, Q, R, Cx, Cu, cn, dt, m, l, family):
    epsilon = 1e-6
    pt = [m, l]
    strategies = [np.concatenate([sol["Psi"].flatten(), sol["Pi"].flatten()]) for sol in family]

    gradient = [approx_fprime(pt, generalizationFunction, epsilon,[A, B, H, Q, R, Cx, Cu, cn, dt, strategy]) for strategy in strategies]

    hessian = [Hessian(lambda params: generalizationFunction(params, [A, B, H, Q, R, Cx, Cu, cn, dt, strategy]))(pt) for strategy in strategies]

    for i, h in enumerate(hessian):
        if np.abs(np.linalg.det(h)) < epsilon:
            hessian[i] += epsilon * np.eye(2)

    nat_grad = [np.dot(np.linalg.inv(hessian[i]), gradient[i]) for i in range(len(family))]
    nat_norm = [np.linalg.norm(grad) for grad in nat_grad]

    return nat_norm

# ===================================================================================================
# Compute characteristics of solution family
# Args:
#   (A, B, H, Q, R), world model parameters
#   (Cx, Cu, cn), loss function parameters
#   dt, timestep (in seconds) between states
#   (m, l) real mass and arm length parameters
#   family, solution set
# Output:
#   pandas dataset with columns:
#   ["id", "family param", "eig[Pi]_real_1", "eig[Pi]_real_2", "eig[Pi]_im_1", "eig[Pi]_im_2",
#                   "observation scaling", "family param (deg)", "local sensitivity"]
# ===================================================================================================
def characterizeFamily(A, B, H, Q, R, Cx, Cu, cn, dt, m, l, family):
    degrees = np.linspace(0, 360, len(family))
    nat_norm = getGeneralization(A, B, H, Q, R, Cx, Cu, cn, dt, m, l, family)

    data = []
    for i, sol in enumerate(family):
        controller_dynamics = np.sort_complex(np.linalg.eigvals(sol['Pi']))
        observation_scaling = np.linalg.det(sol["Psi"] @ sol["Psi"].T)
        data.append([i, degrees[i],
                     np.real(controller_dynamics[0]), np.real(controller_dynamics[1]),
                     np.imag(controller_dynamics[0]), np.imag(controller_dynamics[1]),
                     observation_scaling,
                     nat_norm[i]
                     ])

    return pd.DataFrame(data, columns=["id", "deg", "rPi1", "rPi2", "iPi1", "iPi2", "obs_scaling", "sensitivity_loss"])
