import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap

import seaborn as sns
from typing import Optional
from dataclasses import dataclass

# ====================================================================================
# Constant colors
# ====================================================================================
# Color for baseline performance
baseline_color = [0/255, 0/255, 0/255]

# Colors for solution family
palette = sns.color_palette("hls", 360)
cmap = ListedColormap(palette)
drone_colors = [cmap(i / 360) for i in range(360)]

# Colors for Cart Pole
cart_pole_colors = [drone_colors[-1], drone_colors[180]]
cart_color = [9/255, 97/255, 133/255]
wheels_color = [0/255, 0/255, 0/255]
pole_color = [232/255, 187/255, 10/255]

# ====================================================================================
# Plot state-space trajectories.
# Args:
#   s, numpy array of shape (horizon, state dim, N) -> [x, x_dot, theta, theta_dot]
#   state_dimensions, list of variables (state components)
#   start, time stamp at which visualization starts (in seconds)
#   stop, time stamp at which visualization ends (in seconds)
#   dt, timestep (in seconds) between states
#   sys, stem to visualize (cart_pole or drone)
#   figsize (optional) size of figure
# Output:
#   [figure, axes]
# ====================================================================================
def state_space_trajectories(s, labels, start, stop, dt, sys='', id_color=0, figsize=(12, 5), figures=[]):

    start, stop = int(start/dt), int(stop/dt)
    horizon, dim, N = s.shape
    start = max(0, start)
    stop = min(horizon, stop)

    # Create a new figure with baseline behavior
    if len(figures) == 0:
        fig, axs = plt.subplots(1, int(dim/2), figsize=figsize)
        color = baseline_color
    # Plot frugal behavior over existing figure
    else:
        fig, axs = figures
        if sys == 'cart_pole':
            color = cart_pole_colors[id_color]
        else:
            color = drone_colors[id_color]

    pairs = list(zip(range(0, dim-1, 2), range(1, dim,   2)))
    for idx, pair in enumerate(pairs):
        for tau in range(N):
            x = s[start:stop, pair[0], tau]
            y = s[start:stop, pair[1], tau]
            axs[idx].plot(x, y, color=color, alpha=.1)

        axs[idx].set_xlabel(labels[pair[0]])
        axs[idx].set_ylabel(labels[pair[1]])
        axs[idx].axis('equal')
        axs[idx].grid(False)

    return [fig, axs]

# ====================================================================================
# Plot action sequence (as a function of time)
# Args:
#   a, numpy array of shape (horizon, action dim, N)
#   members_to_viz, list of family members (id) to visualize
#   sys, stem to visualize (cart_pole or drone)
#   start, time stamp at which visualization starts (in seconds)
#   stop, time stamp at which visualization ends (in seconds)
#   dt, timestep (in seconds) between states
# Output:
#   None
# ====================================================================================
def action_trajectories(a, members_to_viz, sys, start, stop, dt):

    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    start, stop = int(start/dt), int(stop/dt)

    if sys == 'cart_pole':
        colors = [baseline_color] + cart_pole_colors
    else:
        colors = [baseline_color] + [drone_colors[i] for i in members_to_viz]

    time_stamps = np.linspace(start, stop, a[0][start:stop].shape[0]) * dt
    for i, tau in enumerate(a):
        color = colors[i]
        axs.plot(time_stamps, tau[start:stop], color=color)

    axs.grid(False)
    axs.set_xlabel('Time (secs)')
    axs.set_ylabel('Acceleration changes')
    plt.show()

# ====================================================================================
# Plot characteristics of family members
# Args:
#   df, pandas DataFrame with characteristics:
#           controller's baseline dynamics rPi, iPi
#           observation scaling
#           sensitivity
#   figsize (optional) size of figure
# Output:
#   None
# ====================================================================================
def props_family(df, figsize=(6, 10)):
    plt.figure(figsize=figsize)
    ax = plt.subplot(4, 1, 1)
    s = 1
    df.plot(x="id", y="rPi1", kind="scatter", s=11, grid=False, ax=ax, c="deg", cmap=cmap, colorbar=False)
    df.plot(x="id", y="rPi2", kind="scatter", s=11, grid=False, ax=ax, c="deg", cmap=cmap, colorbar=False)
    df.plot(x="id", y="iPi1", kind="scatter", s=s, grid=False, ax=ax, c="deg", cmap=cmap, colorbar=False)
    df.plot(x="id", y="iPi2", kind="scatter", s=s, grid=False, ax=ax, c="deg", cmap=cmap, colorbar=False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("base dynamics (control)")

    ax = plt.subplot(4, 1, 2)
    df.plot(x="id", y="obs_scaling", kind="scatter", c="deg", cmap=cmap, grid=False, ax=ax, colorbar=False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("scaling (inference)")

    ax = plt.subplot(4, 1, 3)
    df.plot(x="id", y="sensitivity_loss", kind="scatter", c="deg", cmap=cmap, grid=False, ax=ax, colorbar=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("family parameter (theta)")
    ax.set_ylabel("sensitivity")

    plt.show()

# ====================================================================================
# Display legend denoting colors of special family members
# Args:
#   list, list with members (id) to visualize in future plots
#   labels, name of each strategy
# Output:
#   None
# ====================================================================================
def displaySpecialMembers(list, labels):
    # When solutions are discrete, select two distinctive colors
    if len(list) == 2:
        list = [-1, 180]

    # positions for three bullets (y=0 to keep them on one line)
    xs = [0, 2, 4][:len(list)]
    ys = [0, 0, 0][:len(list)]

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.scatter(xs, ys, s=400, c=[drone_colors[i] for i in list], marker='o')
    for x, y, label in zip(xs, ys, labels):
        ax.text(x + 0.17, y, label, va='center', ha='left', fontsize=11)

    ax.set_xlim(-0.3, 5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')

    plt.show()

# ====================================================================================
# Render animation of CartPole given a trajectory of states previously computed
# ====================================================================================
@dataclass
class CartPoleParams:
    cart_width: float = 0.4      # meters
    cart_height: float = 0.2     # meters
    pole_length: float = 1.0     # meters (pivot to tip)
    wheel_radius: float = 0.05   # meters
    track_half_width: float = 3  # half-length of visible track (meters)
    angle_from_vertical: bool = True  # theta=0 => upright

class CartPoleRenderer:
    def __init__(self, params: CartPoleParams = CartPoleParams()):
        self.p = params

    def _pole_tip(self, x, theta, cart_y):

        # Compute pole tip (px, py) from cart center (x, cart_y) and pole angle theta.
        # If angle_from_vertical=True: theta=0 is upright; CCW positive.
        px = x + self.p.pole_length * np.sin(theta)
        py = cart_y + self.p.pole_length * np.cos(theta)
        return px, py

    # Animate a given trajectory.
    def animate(self, states: np.array, dt: float,
                save_path: Optional[str] = None,
                dpi: int = 120,
                realtime: bool = False):
        """
        Args:
            states: numpy array of shape (T, 4) -> [x, x_dot, theta, theta_dot]
            dt: timestep in seconds between consecutive states
            save_path: if provided, saves animation as a .gif
            dpi: output resolution for saving
            realtime: if True, uses interval=dt*1000; else a faster preview.
        """

        T = states.shape[0]
        x = states[:, 0]
        theta = states[:, 2]

        p = self.p
        # Scene layout
        ground_y = 0.0
        cart_y = ground_y + p.wheel_radius + p.cart_height / 2.0

        # Figure & axes
        fig, ax = plt.subplots(figsize=(8, 4.0))
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-p.track_half_width, p.track_half_width)
        # Vertical limits: add room above for pole
        ax.get_yaxis().set_visible(False)
        y_max = cart_y + p.pole_length + 0.3
        ax.set_ylim(-0.1, max(y_max, 0.6))

        # Dynamic artists
        # Cart body (as a rectangle built from a Line2D for simplicity)
        cart_left = x[0] - p.cart_width / 2
        cart_right = x[0] + p.cart_width / 2
        cart_top = cart_y + p.cart_height / 2
        cart_bot = cart_y - p.cart_height / 2
        cart_outline_x = [cart_left, cart_right, cart_right, cart_left, cart_left]
        cart_outline_y = [cart_bot,  cart_bot,   cart_top,   cart_top,  cart_bot]
        (cart_line,) = ax.plot(cart_outline_x, cart_outline_y, linewidth=2, color=cart_color)

        # Wheels (circles approximated by polygons)
        theta_circle = np.linspace(0, 2*np.pi, 40)
        wheel_offsets = np.array([-p.cart_width/3, p.cart_width/3])
        wheel_lines = []
        for w in wheel_offsets:
            wx = x[0] + w + p.wheel_radius * np.cos(theta_circle)
            wy = ground_y + p.wheel_radius * np.sin(theta_circle)
            (wl,) = ax.plot(wx, wy, linewidth=1.5, color=wheels_color)
            wheel_lines.append(wl)

        # Pole
        px0, py0 = self._pole_tip(x[0], theta[0], cart_y)
        (pole_line,) = ax.plot([x[0], px0], [cart_y, py0], linewidth=2, color=pole_color)

        # Pivot marker
        (pivot_point,) = ax.plot([x[0]], [cart_y], marker='o', color=pole_color)

        # Time text
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")

        def update(frame):
            xc = x[frame]
            th = theta[frame]

            # Update cart rectangle
            left = xc - p.cart_width / 2
            right = xc + p.cart_width / 2
            cart_line.set_data(
                [left, right, right, left, left],
                [cart_bot, cart_bot, cart_top, cart_top, cart_bot]
            )

            # Update wheels
            for i, w in enumerate(wheel_offsets):
                wx = xc + w + p.wheel_radius * np.cos(theta_circle)
                wy = ground_y + p.wheel_radius * np.sin(theta_circle)
                wheel_lines[i].set_data(wx, wy)

            # Update pole
            px, py = self._pole_tip(xc, th, cart_y)
            pole_line.set_data([xc, px], [cart_y, py])

            # Update pivot
            pivot_point.set_data([xc], [cart_y])

            # Update time display
            time_text.set_text(f"t = {frame*dt:.2f} s")

            return cart_line, pole_line, pivot_point, *wheel_lines, time_text

        interval_ms = max(1, int(dt * 1000)) if realtime else 20
        anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)

        if save_path is not None:
            anim.save(save_path, writer=PillowWriter(fps=int(1.0/dt) if realtime else 50), dpi=dpi)
        else:
            plt.show()

        plt.close(fig)

# ====================================================================================
# Render animation of Planar Drone given a list of 4 trajectories of states
# NOTE: The last trajectury should be baseline performance
# ====================================================================================
@dataclass
class SimpleDroneParams:
    half_len: float = 1.     # half of the drone line length (meters)
    pad: float = 2.0         # padding around auto-fit view (meters)

class SimpleMultiDroneRenderer:
    def __init__(self, params: SimpleDroneParams = SimpleDroneParams()):
        self.p = params

    def _normalize_states(self, states):
        """
        Accepts:
          - list/tuple of 4 arrays, each (T, 6)
          - numpy array shaped (4, T, 6) or (T, 4, 6)
        Returns:
          arr of shape (4, T, 6)
        """
        if isinstance(states, (list, tuple)):
            if len(states) != 4:
                raise ValueError("Provide exactly 4 state trajectories.")
            Ts = [np.asarray(s) for s in states]
            for i, s in enumerate(Ts):
                if s.ndim != 2 or s.shape[1] != 6:
                    raise ValueError(f"Trajectory {i} must have shape (T, 6); got {s.shape}.")
            T0 = Ts[0].shape[0]
            if any(s.shape[0] != T0 for s in Ts):
                raise ValueError("All 4 trajectories must have the same horizon T.")
            return np.stack(Ts, axis=0)  # (4, T, 6)

        arr = np.asarray(states)
        if arr.ndim == 3 and arr.shape == (4, arr.shape[1], 6):
            return arr
        if arr.ndim == 3 and arr.shape[2] == 6 and arr.shape[1] == 4:
            return np.transpose(arr, (1, 0, 2))  # (T,4,6)->(4,T,6)
        raise ValueError(
            "states must be a list of 4 (T,6) arrays, or an array shaped (4,T,6) or (T,4,6)."
        )

    def animate(self, states, dt: float, identities: Optional[np.array] = np.arange(4),
                save_path: Optional[str] = None,
                dpi: int = 120,
                realtime: bool = False):
        """
        Animate 4 planar drones as oriented line segments.

        Args:
            states: list of 4 (T,6) arrays OR array shaped (4,T,6) or (T,4,6)
                    State order: [x, x_dot, y, y_dot, tilt, tilt_dot]
            dt: timestep (s) between states
            identities: list of colors to identify drones (match to family)
            save_path: .gif or .mp4 to save; None -> show
            dpi: output resolution if saving
            realtime: if True, playback interval = dt*1000; else ~50 fps preview
            show_axes: if True, keep axes; else hide box for a clean look
            show_ground: draw a faint baseline at y=0 if visible
        """
        S = self._normalize_states(states)          # (4, T, 6)
        N, T, _ = S.shape
        if N != 4:
            raise RuntimeError("Internal error: expected 4 trajectories after normalization.")

        X = S[:, :, 0]
        Y = S[:, :, 2]
        TH = S[:, :, 4]

        # Fit bounds over all drones with padding
        pad = self.p.pad
        xmin, xmax = float(np.min(X) - pad), float(np.max(X) + pad)
        ymin, ymax = float(np.min(Y) - pad), float(np.max(Y) + pad)
        ymin = min(ymin, -0.5)
        ymax = max(ymax,  0.8)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Hide the box and ticks for a clean animation-only look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # One line + point per drone
        body_lines = []
        com_points = []
        current_colors = [drone_colors[i] for i in identities] + [baseline_color]
        for idx in range(4):
            (ln,) = ax.plot([], [], linewidth=2, color=current_colors[idx])    # body line
            (pt,) = ax.plot([], [], marker='o', markersize=4, color=current_colors[idx])  # center-of-mass dot
            body_lines.append(ln)
            com_points.append(pt)

        time_text = ax.text(0.3, 0.96, "", transform=ax.transAxes, ha="left", va="top")

        L = self.p.half_len

        def update(i):
            for k in range(4):
                xc, yc, th = X[k, i], Y[k, i], TH[k, i]
                c, s = np.cos(th), np.sin(th)
                # endpoints of the line segment in world coordinates
                x1, y1 = xc - L * c, yc - L * s
                x2, y2 = xc + L * c, yc + L * s
                body_lines[k].set_data([x1, x2], [y1, y2])
                com_points[k].set_data([xc], [yc])
            time_text.set_text(f"t = {i*dt:.2f} s")
            return (*body_lines, *com_points, time_text)

        interval_ms = max(1, int(dt * 1000)) if realtime else 20
        anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)

        if save_path is not None:
            anim.save(save_path, writer=PillowWriter(fps=int(1.0/dt) if realtime else 50), dpi=dpi)
        else:
            plt.show()

        plt.close(fig)
        return anim