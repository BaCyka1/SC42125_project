import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


class Simulation:
    def __init__(self, drone, mpc_controller, g=9.81, dt=0.01, linear=True, LQR=False):
        self.drone=drone
        self.MPC_controller = mpc_controller
        self.g=g
        self.dt=dt
        # Simulation type:
        self.linear = linear
        self.LQR = LQR
        self.state_history = []
        self.input_history = []
        self.t = 0
        self.F1, self.F2 = 0, 0

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

        if not self.linear:
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
                        -m_d * np.cos(psi) - m_l * np.sin(psi) * np.sin(theta) * np.cos(theta) + m_l * np.cos(psi) * np.cos(
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

            new_state = np.array([x_new, y_new, psi_new, theta_new, vx_new, vy_new, psi_dot_new, theta_dot_new])

        else:
            x_dot = self.drone.A_c @ self.drone.state + self.drone.B_c @ np.array([F1, F2])
            x_dot[5] -= self.g # Add continuous gravity effect directly
            new_state = self.drone.state + x_dot * self.dt

        self.t += self.dt

        self.drone.update_state(new_state)

    # Disclaimer: Animation code written with the help of GPT-4o
    def animation_step(self, frame):
        # Get control inputs from the controller
        # Choose between MPC and LQR
        if not self.LQR:
            if self.t % self.MPC_controller.timestep < 0.01:
                self.F1, self.F2 = self.MPC_controller.compute_control(x_0=self.drone.state)

        else:
            K = self.MPC_controller.K
            x = self.drone.state
            x_ref = self.MPC_controller.x_ref
            u_ref = self.MPC_controller.u_ref
            self.F1, self.F2 = - K @ (x - x_ref) + u_ref

        # Update states with control input and dynamics
        self.physics_step(self.F1, self.F2)

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

        # Compute load position
        load_x = x + self.drone.L_l * np.sin(theta)
        load_y = y - self.drone.L_l * np.cos(theta)

        # Update pendulum
        self.rod_line.set_data([x, load_x], [y, load_y])
        self.load_patch.center = (load_x, load_y)

        # Append inputs to history
        self.state_history.append(self.drone.state.copy())
        self.input_history.append(np.array([self.F1, self.F2]))

        return self.body_patch, self.thruster_left, self.thruster_right, self.arm_left, self.arm_right, self.load_patch, self.rod_line

    def plot_states(self):
        if not self.state_history:
            print("No history to plot.")
            return

        state_array = np.array(self.state_history)
        input_array = np.array(self.input_history)

        labels = ['x', 'y', 'psi', 'theta', 'vx', 'vy', 'psi_dot', 'theta_dot', "u_1", "u_2"]
        fig, axs = plt.subplots(5, 2, figsize=(12, 10))
        axs = axs.flatten()
        t = np.arange(len(state_array)) * self.dt

        for i in range(8):
            axs[i].plot(t, state_array[:, i])
            axs[i].set_title(labels[i])
            axs[i].set_xlabel("Time [s]")
            axs[i].grid(True)

        for i in (8, 9):
            axs[i].plot(t, input_array[:, i-8])
            axs[i].set_title(labels[i])
            axs[i].set_xlabel("Time [s]")
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_lyapunov_decrease(self):
        if not self.state_history or len(self.state_history) < 2:
            print("Not enough history to plot Lyapunov decrease.")
            return

        state_array = np.array(self.state_history)
        input_array = np.array(self.input_history) # Recorded at sim_dt

        sim_dt = self.dt
        mpc_dt = self.MPC_controller.timestep

        # Calculate the number of full MPC intervals available in the data
        steps_per_mpc_interval = int(round(mpc_dt / sim_dt))
        num_mpc_intervals = (len(state_array) - 1) // steps_per_mpc_interval

        # Check if controller references exist (OTS must have run successfully)
        x_ref = self.MPC_controller.x_ref
        u_ref = self.MPC_controller.u_ref
        P = self.MPC_controller.P
        Q = self.MPC_controller.Q
        R = self.MPC_controller.R

        negative_running_costs = []
        terminal_costs_k = []       # V_f(x_k) at MPC times
        terminal_costs_k_plus_1 = [] # V_f(x_{k+1}) at MPC times

        # Iterate through MPC intervals k = 0 to num_mpc_intervals - 1
        for k in range(num_mpc_intervals):
            idx_k = k * steps_per_mpc_interval       # Index in state/input_array for start of interval k
            idx_k_plus_1 = (k + 1) * steps_per_mpc_interval # Index for start of interval k+1

            x_k = state_array[idx_k]
            x_k_plus_1 = state_array[idx_k_plus_1]

            # Get the input applied at the start of interval k (it was held constant)
            u_k = input_array[idx_k]

            # Calculate Lyapunov components
            delta_x_k = x_k - x_ref
            delta_x_k_plus_1 = x_k_plus_1 - x_ref
            delta_u_k = u_k - u_ref

            # Running Cost: l(x_k, u_k)
            running_cost_k = 0.5 * (delta_x_k.T @ Q @ delta_x_k + delta_u_k.T @ R @ delta_u_k)
            negative_running_costs.append(-running_cost_k)

            # Terminal cost at start of interval k: V_f(x_k)
            term_cost_k = delta_x_k.T @ P @ delta_x_k
            terminal_costs_k.append(term_cost_k)

            # Terminal cost at start of interval k+1: V_f(x_{k+1})
            term_cost_k_plus_1 = delta_x_k_plus_1.T @ P @ delta_x_k_plus_1
            terminal_costs_k_plus_1.append(term_cost_k_plus_1)

        # Difference in terminal costs: V_f(x_{k+1}) - V_f(x_k) over MPC intervals
        terminal_diffs = np.array(terminal_costs_k_plus_1) - np.array(terminal_costs_k)

        # Time vector for plotting (corresponding to MPC steps k=0 to num_mpc_intervals-1)
        t = np.arange(len(terminal_diffs)) * mpc_dt # Use MPC timestep

        plt.figure(figsize=(10, 6))
        # Only plot if we have data
        if len(t) > 0:
             plt.plot(t, terminal_diffs, label="$V_f(x_{k+1}) - V_f(x_k)$", marker='x', linestyle='-', color='blue')
             plt.plot(t, negative_running_costs, label="$-l(x_k, u_k)$", marker='o', linestyle='--', color='orange')

             # Add a zero line for reference
             plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)

        plt.xlabel("Time [s]")
        plt.ylabel("Value")
        plt.title("Lyapunov Decrease: (MPC Timesteps): $V_f(x_{k+1}) - V_f(x_k)$ vs $-l(x_k, u_k)$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_simulation(self, frames=10):
        ani = animation.FuncAnimation(self.fig, self.animation_step, frames=frames, interval=self.dt * 1000, blit=False, repeat=False)
        # ani.save("Animation.gif", fps=30)
        plt.show()
        self.plot_states()
        # self.plot_lyapunov_decrease()