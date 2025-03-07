import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

# Simulation Parameters
dt = 0.05  # Time step (seconds)
m_d = 1.0  # Drone mass
m_l = 0.1  # Load mass
g = 9.81  # Gravity (m/s²)
d = 2.0  # Distance between rotors
I_d = 0.8  # Drone moment of inertia
L_d = d / 2  # Distance from drone COM to each thruster
L_l = 1.5  # Length of the massless rod suspending the load

# Total mass
m_tot = m_d + m_l

# Baseline thrust for hovering: each rotor must produce half the total weight
baseline = m_tot * g / 2.0
delta = 1.0  # Thrust adjustment step

# Initial Drone State
# State: [x, y, psi, theta, vx, vy, omega, theta_dot]
state = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
F1, F2 = baseline, baseline

# Matplotlib Setup
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D Drone with Suspended Load Simulation')

# Create sprites
body_radius = 0.5  # Drone body radius
thruster_size = (0.4, 0.4)  # Thruster width & height
arm_length = L_d  # Arm length (from COM to thruster)

# Drone main body (circle)
body_patch = Circle((0, 0), body_radius, fc='grey', ec='black')

# Arms (lines from COM to thrusters)
arm_left = Line2D([0, 0], [0, 0], lw=4, color='black')
arm_right = Line2D([0, 0], [0, 0], lw=4, color='black')

# Thrusters (rectangles)
thruster_left = Rectangle((-arm_length - thruster_size[0] / 2, -thruster_size[1] / 2),
                          thruster_size[0], thruster_size[1], fc='black', ec='black')
thruster_right = Rectangle((arm_length - thruster_size[0] / 2, -thruster_size[1] / 2),
                           thruster_size[0], thruster_size[1], fc='black', ec='black')

# Load (suspended mass) as a small circle
load_radius = 0.2
load_patch = Circle((0, 0), load_radius, fc='orange', ec='black')

# Rod connecting drone and load
rod_line = Line2D([0, 0], [0, 0], lw=2, color='gray')

# Add patches and lines to the plot
ax.add_patch(body_patch)
ax.add_patch(thruster_left)
ax.add_patch(thruster_right)
ax.add_patch(load_patch)
ax.add_line(arm_left)
ax.add_line(arm_right)
ax.add_line(rod_line)


# Drone + Load Dynamics Function (Corrected)
def update_state(state, F1, F2, dt):
    """
    Update the state using the coupled drone-load dynamics.
    State: [x, y, psi, theta, vx, vy, omega, theta_dot]
    """
    x, y, psi, theta, vx, vy, psi_dot, theta_dot = state

    F_T = F1 + F2
    torque = L_d * (F2 - F1)

    # Solve for translational accelerations


    D_xy = -m_d**2 + m_d*m_l*np.sin(theta)**2 + m_d*m_l*np.cos(theta)**2 -2*m_d*m_l + m_l**2 * np.sin(theta)**2 + m_l**2 * np.cos(theta)**2 - m_l**2

    ddx = F_T * (m_d*np.sin(psi) - m_l*np.sin(psi)*np.sin(theta)**2 + m_l*np.sin(psi) - m_l*np.sin(theta)*np.cos(psi)*np.cos(theta))
    ddx += L_l * (-m_d*m_l*np.sin(theta)*theta_dot**2 + m_l**2 *np.sin(theta)**3 * theta_dot**2)
    ddx += L_l * (m_l**2 * np.sin(theta) * np.cos(theta)**2 * theta_dot**2 - m_l**2 * np.sin(theta) * theta_dot**2)
    ddx /= D_xy

    ddy = F_T * (-m_d * np.cos(psi) + m_l*np.sin(psi)*np.sin(theta)*np.cos(theta) + m_l * np.cos(psi) * np.cos(theta)**2 - m_l*np.cos(psi))
    ddy += g * (m_d**2 - m_d*m_l*np.sin(theta)**2 - m_d*m_l*np.cos(theta)**2 + 2*m_d*m_l - m_l**2 * np.sin(theta)**2 - m_l**2 * np.cos(theta)**2 + m_l**2)
    ddy += L_l * (m_d*m_l*np.cos(theta)*theta_dot**2 - m_l**2 * np.sin(theta)**2 * np.cos(theta) * theta_dot**2 - m_l**2 * np.cos(theta)**3 * theta_dot**2 + m_l**2 * np.cos(theta) * theta_dot**2)
    ddy /= D_xy

    D_theta = -L_l * m_d + L_l * m_l * np.sin(theta)**2 + L_l * m_l * np.cos(theta)**2 - L_l * m_l

    # Pendulum (load) dynamics via the constraint:
    ddtheta = F_T * ( - np.sin(psi)*np.cos(theta) + np.cos(psi)*np.sin(theta)) / D_theta

    # Drone rotational acceleration
    ddpsi = torque / I_d

    # Euler integration
    vx_new = vx + ddx * dt
    vy_new = vy + ddy * dt
    psi_dot_new = psi_dot + ddpsi * dt
    theta_dot_new = theta_dot + ddtheta * dt

    x_new = x + vx * dt
    y_new = y + vy * dt
    psi_new = psi + psi_dot * dt
    theta_new = theta + theta_dot * dt

    return np.array([x_new, y_new, psi_new, theta_new, vx_new, vy_new, psi_dot_new, theta_dot_new])


# Animation Update Function
def animate(frame):
    global state, F1, F2
    state = update_state(state, F1, F2, dt)
    x, y, psi, theta, _, _, _, _ = state

    # Rotation matrix for drone orientation psi
    R = np.array([[np.cos(psi), -np.sin(psi)],
                  [np.sin(psi), np.cos(psi)]])

    # Update drone body position
    body_patch.center = (x, y)

    # Compute thruster offsets in the drone’s body frame
    offsets = np.array([[-L_d, 0], [L_d, 0]])
    rotated_offsets = (R @ offsets.T).T
    for thruster, offset in zip([thruster_left, thruster_right], rotated_offsets):
        thruster_center = np.array([x, y]) + offset
        corner = thruster_center - np.array([thruster_size[0] / 2, thruster_size[1] / 2])
        thruster.set_xy(corner)
        thruster.angle = np.degrees(psi)

    # Update arms (lines from COM to thrusters)
    arm_left.set_data([x, x + rotated_offsets[0, 0]], [y, y + rotated_offsets[0, 1]])
    arm_right.set_data([x, x + rotated_offsets[1, 0]], [y, y + rotated_offsets[1, 1]])

    # Compute load position from drone position and rod geometry
    load_x = x + L_l * np.sin(theta)
    load_y = y - L_l * np.cos(theta)
    load_patch.center = (load_x, load_y)

    # Update rod (line) connecting drone and load
    rod_line.set_data([x, load_x], [y, load_y])

    return body_patch, thruster_left, thruster_right, arm_left, arm_right, load_patch, rod_line


# Interactive Control
def on_key_press(event):
    global F1, F2, baseline, delta
    if event.key == 'up':  # Increase both forces → Ascend
        F1 += delta
        F2 += delta
    elif event.key == 'down':  # Decrease both forces → Descend
        F1 -= delta
        F2 -= delta
    elif event.key == 'left':  # Counterclockwise rotation
        F1 -= delta
        F2 += delta
    elif event.key == 'right':  # Clockwise rotation
        F1 += delta
        F2 -= delta


def on_key_release(event):
    global F1, F2, baseline
    F1 = baseline
    F2 = baseline

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# Run Animation
ani = animation.FuncAnimation(fig, animate, frames=600, interval=dt * 1000, blit=False)
plt.show()
