import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


class Simulation:
    def __init__(self, drone, mpc_controller, g=9.81, dt=0.05):
        self.drone=drone
        self.MPC_controller = mpc_controller
        self.g=g
        self.dt=dt

        # Matplotlib Setup
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(0, 15)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('2D Drone with Suspended Load')

        body_radius = 0.5  # Drone body radius
        self.thruster_size = (0.4, 0.4)  # Thruster width & height

        # Drone main body (circle)
        self.body_patch = Circle((0, 0), body_radius, fc='grey', ec='black')

        # Arms (lines from COM to thrusters)
        self.arm_left = Line2D([0, 0], [0, 0], lw=4, color='black')
        self.arm_right = Line2D([0, 0], [0, 0], lw=4, color='black')

        # Thrusters (rectangles)
        self.thruster_left = Rectangle((-self.drone.L_d - self.thruster_size[0] / 2, -self.thruster_size[1] / 2),
                                  self.thruster_size[0], self.thruster_size[1], fc='black', ec='black', rotation_point="center")
        self.thruster_right = Rectangle((self.drone.L_d - self.thruster_size[0] / 2, -self.thruster_size[1] / 2),
                                   self.thruster_size[0], self.thruster_size[1], fc='black', ec='black', rotation_point="center")

        # Load (suspended mass) as a small circle
        load_radius = 0.2
        self.load_patch = Circle((0, 0), load_radius, fc='orange', ec='black')

        # Rod connecting drone and load
        self.rod_line = Line2D([0, 0], [0, 0], lw=2, color='black')

        # Add patches and lines to the plot
        self.ax.add_patch(self.body_patch)
        self.ax.add_patch(self.thruster_left)
        self.ax.add_patch(self.thruster_right)
        self.ax.add_line(self.rod_line)
        self.ax.add_line(self.arm_left)
        self.ax.add_line(self.arm_right)
        self.ax.add_patch(self.load_patch)

    def physics_step(self, F1, F2):
        """
        Update the state using the coupled drone-load dynamics.
        State: [x, y, psi, theta, vx, vy, omega, theta_dot]
        """
        # Get state from drone object
        x, y, psi, theta, vx, vy, psi_dot, theta_dot = self.drone.state

        m_d, m_l, I_d, L_d, L_l = self.drone.constants

        # Calculate total force and torque caused by thust
        F_T = F1 + F2
        torque = L_d * (F2 - F1)

        # Solve for translational accelerations
        # Denominator of translational accelerations
        D_xy = -m_d ** 2 + m_d * m_l * np.sin(theta) ** 2 + m_d * m_l * np.cos(
            theta) ** 2 - 2 * m_d * m_l + m_l ** 2 * np.sin(theta) ** 2 + m_l ** 2 * np.cos(theta) ** 2 - m_l ** 2

        # x acceleration in inertial frame
        ddx = F_T * (m_d * np.sin(psi) - m_l * np.sin(psi) * np.sin(theta) ** 2 + m_l * np.sin(psi) - m_l * np.sin(
            theta) * np.cos(psi) * np.cos(theta))
        ddx += L_l * (-m_d * m_l * np.sin(theta) * theta_dot ** 2 + m_l ** 2 * np.sin(theta) ** 3 * theta_dot ** 2)
        ddx += L_l * (m_l ** 2 * np.sin(theta) * np.cos(theta) ** 2 * theta_dot ** 2 - m_l ** 2 * np.sin(
            theta) * theta_dot ** 2)
        ddx /= D_xy

        # y acceleration in inertial frame
        ddy = F_T * (
                    -m_d * np.cos(psi) + m_l * np.sin(psi) * np.sin(theta) * np.cos(theta) + m_l * np.cos(psi) * np.cos(
                theta) ** 2 - m_l * np.cos(psi))
        ddy += self.g * (m_d ** 2 - m_d * m_l * np.sin(theta) ** 2 - m_d * m_l * np.cos(
            theta) ** 2 + 2 * m_d * m_l - m_l ** 2 * np.sin(theta) ** 2 - m_l ** 2 * np.cos(theta) ** 2 + m_l ** 2)
        ddy += L_l * (m_d * m_l * np.cos(theta) * theta_dot ** 2 - m_l ** 2 * np.sin(theta) ** 2 * np.cos(
            theta) * theta_dot ** 2 - m_l ** 2 * np.cos(theta) ** 3 * theta_dot ** 2 + m_l ** 2 * np.cos(
            theta) * theta_dot ** 2)
        ddy /= D_xy

        # Denominator of theta acceleration
        D_theta = -L_l * m_d + L_l * m_l * np.sin(theta) ** 2 + L_l * m_l * np.cos(theta) ** 2 - L_l * m_l

        # Pendulum (load) dynamics:
        ddtheta = F_T * (- np.sin(psi) * np.cos(theta) + np.cos(psi) * np.sin(theta)) / D_theta

        # Drone rotational acceleration
        ddpsi = torque / I_d

        # Euler integration
        vx_new = vx + ddx * self.dt
        vy_new = vy + ddy * self.dt
        psi_dot_new = psi_dot + ddpsi * self.dt
        theta_dot_new = theta_dot + ddtheta * self.dt

        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        psi_new = psi + psi_dot * self.dt
        theta_new = theta + theta_dot * self.dt

        self.drone.update_state(np.array([x_new, y_new, psi_new, theta_new, vx_new, vy_new, psi_dot_new, theta_dot_new]))

    def animation_step(self, frame):
        target_state1 = np.array([8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Get control inputs from the controller
        F1, F2 = self.MPC_controller.compute_control(current_state=self.drone.state, target_state=target_state1)

        # Update states with control input and dynamics
        self.physics_step(F1, F2)

        # Unpack current states
        x, y, psi, theta, _, _, _, _ = self.drone.state

        # Rotation matrix for drone orientation psi
        R = np.array([[np.cos(psi), -np.sin(psi)],
                      [np.sin(psi), np.cos(psi)]])

        # Update drone body position
        self.body_patch.center = (x, y)

        # Compute thruster offsets in the drone’s body frame
        offsets = np.array([[-self.drone.L_d, 0], [self.drone.L_d, 0]])
        rotated_offsets = (R @ offsets.T).T
        for thruster, offset in zip([self.thruster_left, self.thruster_right], rotated_offsets):
            thruster_center = np.array([x, y]) + offset
            corner = thruster_center - np.array([self.thruster_size[0] / 2, self.thruster_size[1] / 2])
            thruster.set_xy(corner)
            thruster.angle = np.degrees(psi)

        # Update arms (lines from COM to thrusters)
        self.arm_left.set_data([x, x + rotated_offsets[0, 0]], [y, y + rotated_offsets[0, 1]])
        self.arm_right.set_data([x, x + rotated_offsets[1, 0]], [y, y + rotated_offsets[1, 1]])

        # Compute load position from drone position and rod geometry
        load_x = x + self.drone.L_l * np.sin(theta)
        load_y = y - self.drone.L_l * np.cos(theta)

        # Update rod (line) connecting drone and load
        self.rod_line.set_data([x, load_x], [y, load_y])
        self.load_patch.center = (load_x, load_y)
        return self.body_patch, self.thruster_left, self.thruster_right, self.arm_left, self.arm_right, self.load_patch, self.rod_line

    def run_simulation(self):
        ani = animation.FuncAnimation(self.fig, self.animation_step, frames=600, interval=self.dt * 1000, blit=False)
        plt.show()