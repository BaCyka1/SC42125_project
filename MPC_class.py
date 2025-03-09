import numpy as np
import cvxpy as cp

class MPCController:
    def __init__(self, drone, horizon=500, controller_timestep=1/60):
        self.horizon = horizon
        self.timestep = controller_timestep

        g = 9.81
        m_d = drone.m_d
        m_l = drone.m_l
        I_d = drone.I_d
        L_d = drone.L_d
        L_l = drone.L_l
        self.m = m_d + m_l

        # Continuous time A matrix
        self.A_c = np.zeros(shape=(8,8))
        self.A_c[:4, 4:] = np.eye(4)
        self.A_c[4, 2] = - g * (m_d + m_l) / m_d
        self.A_c[4, 3] = g * m_l / m_d
        self.A_c[7, 2] = g * (m_d + m_l) / (m_d * L_l)
        self.A_c[7, 3] = - g * (m_d + m_l) / (m_d * L_l)

        # Continuous time B matrix
        self.B_c = np.zeros(shape=(8, 2))
        self.B_c[5, 0] = 1/(m_d + m_l)
        self.B_c[5, 1] = 1 / (m_d + m_l)
        self.B_c[6, 0] = - L_d / I_d
        self.B_c[6, 1] = L_d / I_d

        # Conversion to discrete time
        self.A = np.eye(8) + self.A_c * self.timestep
        self.B = self.B_c * self.timestep
        self.C = np.eye(8)
        self.D = np.zeros((1, 2))

    def compute_control(self, current_state, target_state):
        # Penalties
        Q = np.diag([10, 10, 5, 1, 1, 1, 1, 1]) # State penalties
        R = 0.01 * np.eye(2) # Control input penalties

        # Hovering target
        u_target = np.array([self.m * 9.81 / 2, self.m * 9.81 / 2])

        # Create the optimization variables
        x = cp.Variable((8, self.horizon + 1))
        u = cp.Variable((2, self.horizon))

        # Constraints and initial cost
        constraints = []
        cost = 0.

        # Initial state
        constraints += [x[:, 0] == current_state.flatten()]

        for n in range(self.horizon):
            cost += (cp.quad_form((x[:,n+1]-target_state),Q)  + cp.quad_form(u[:,n]-u_target, R))
            constraints += [x[:,n+1] == self.A @ x[:,n] + self.B @ u[:,n]]

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(verbose=False) # solver=cp.OSQP

        # Return first timestep of trajectory
        return u[:, 0].value