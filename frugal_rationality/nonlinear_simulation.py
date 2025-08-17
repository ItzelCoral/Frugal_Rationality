import numpy as np

# ===============================================================================================
# This functions simulate the behavior of a nonlinear dynamical system
# Args:
#  env, the environment that will be simulated (cartpole or drone)
#  strategy, the params defining the inference and control policy [G,K,L] or [Pi, Psi]
#  noise_cov, the covariance matrices [Q, R] defining process and observation noise
#  init_s, initial value of state
#  init_u, initial value of action
#  dt, timestep (in seconds) between states
#  horizon, number of time steps to be simulated
#  N, number of simulations to be executed (large value for statistical purposes)
#  lqg, which strategy will be simulated (LQG or frugal)
# Output:
#   s_trajectoy, tensor with shape [horizon, state dimension, N] containing state trajectories
#   u_trajectoy, tensor with shape [horizon, action dimension, N] containing action trajectories
# ===============================================================================================

def nonlinear_sim(env, strategy, noise_cov, init_s, init_u, dt=0.01, horizon=5000, N=1, lqg=False):

    # Generate noise in advance to speed up simulation
    Q, R = noise_cov
    proc_n = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q, [horizon, N]).T
    obs_n = np.random.multivariate_normal(np.zeros(R.shape[0]), R, [horizon, N]).T

    # Initialize current state (s), action (u), and estimate (mu)
    s = np.array([init_s]*N).T
    u = np.array([init_u]*N).T
    mu = np.zeros_like(s)

    # Initialize record of states and actions that the strategy generates
    s_trajectoy = [s.copy()]
    u_trajectoy = [u.copy()]

    # The LQG controller calculates estimates explicitly
    if lqg:
        for i in range(horizon):
            o = s + obs_n[:, :, i]
            mu = strategy["G"] @ mu + strategy["K"] @ o
            u = strategy["L"] @ mu

            s_dot = env.step(s, u)
            s = s + dt * s_dot + proc_n[:, :, i]

            s_trajectoy.append(s.copy())
            u_trajectoy.append(u.copy())

    # A frugal strategy calculates estimates implicitly
    else:
        for i in range(horizon):
            o = s + obs_n[:, :, i]
            u = strategy["Pi"] @ u + strategy["Psi"] @ o

            s_dot = env.step(s, u)
            s = s + dt * s_dot + proc_n[:, :, i]

            s_trajectoy.append(s.copy())
            u_trajectoy.append(u.copy())

    return np.array(s_trajectoy), np.array(u_trajectoy)

# ====================================================================================
# Environments: CartPole and Planar drone
# ====================================================================================
class CartPole:
    def __init__(self, m, M, l, g, d):

        self.m = m  # Mass of the pendulum
        self.M = M  # Mass of the cart
        self.l = l  # Length of the pendulum
        self.g = g  # Acceleration due to gravity, reframed, now up is down so goal is zeros
        self.d = d  # Damping factor

    def step(self, s, u):
        x, v, theta, omega = s
        u = u[0]

        ddv = ((-self.m ** 2 * self.l ** 2 * self.g * np.cos(theta) * np.sin(theta) +
               self.m * self.l ** 2 * (self.m * self.l * omega ** 2 * np.sin(theta) - self.d * v) +
               self.m * self.l ** 2 * u) / (self.m * self.l ** 2 * (self.M + self.m * (1 - np.cos(theta) ** 2))))
        ddomega = (((self.m + self.M) * self.m * self.g * self.l * np.sin(theta) -
                self.m * self.l * np.cos(theta) * (self.m * self.l * omega ** 2 * np.sin(theta) - self.d * v) -
                self.m * self.l * np.cos(theta) * u) / (self.m * self.l ** 2 * (self.M + self.m * (1 - np.cos(theta) ** 2))))

        return np.array([v, ddv, omega, ddomega])


class PlanarDrone:
    def __init__(self, m, l, g, dt):
        self.m = m
        self.l = l
        self.I = 2 * m * l ** 2
        self.g = g
        self.dt = dt

    def step(self, s, u):
        x, dx, y, dy, th, dth = s
        u1, u2 = u

        ddx = -(u1 + u2) * np.sin(th) / self.m
        ddy = (u1 + u2) * np.cos(th) / self.m - self.g
        ddth = self.l * (u1 - u2) / self.I

        return np.array([dx, ddx, dy, ddy, dth, ddth])




