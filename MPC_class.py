import numpy as np
import cvxpy as cp
from control import dare
from scipy.linalg import expm


class MPCController:
    def __init__(self, drone, horizon=500, dt=1/60):
        self.horizon = horizon
        self.timestep = dt

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

        dim_x_d = self.A_c.shape[0]
        dim_u_d = self.B_c.shape[1]

        # Conversion to discrete time
        AB_c = np.zeros((dim_x_d + dim_u_d, dim_x_d + dim_u_d))
        AB_c[:dim_x_d, :dim_x_d] = self.A_c
        AB_c[:dim_x_d, dim_x_d:] = self.B_c
        expm_AB_c = expm(AB_c * dt)
        self.A_d = expm_AB_c[:dim_x_d, :dim_x_d]
        self.B_d = expm_AB_c[:dim_x_d, dim_x_d:]
        print(self.A_d.shape)

        H_x = np.vstack((np.eye(8), np.eye(8) * -1))

        h_x = np.ones(H_x.shape[0]) * 100

        H_u = np.array([[1, 0],
                        [0, 1],
                        [-1, 0],
                        [0, -1]])

        h_u = np.ones(H_u.shape[0]) * 10

        self.problem_constraints = (H_x, H_u, h_x, h_u)

    def compute_control(self, current_state, target_state, gamma=2.18867187):
        # Penalties
        Q = np.diag([10, 10, 5, 10, 1, 1, 1, 5]) # State penalties
        R = np.eye(2) * 0.01 # Control input penalties

        P, _, _ = dare(self.A_d, self.B_d, Q, R)

        # Hovering target
        u_target = np.array([self.m * 9.81 / 2, self.m * 9.81 / 2])

        # Unpack the constraints
        H_x, H_u, h_x, h_u = self.problem_constraints

        # Variables
        x = [cp.Variable(8) for _ in range(self.horizon + 1)]
        u = [cp.Variable(2) for _ in range(self.horizon)]

        # Constraints and initial cost
        constraints = []
        cost = 0.

        # Initial state
        constraints += [x[0] == current_state.flatten()]

        for k in range(self.horizon):
            cost += (cp.quad_form((x[k+1]-target_state),Q)  + cp.quad_form(u[k]-u_target, R))
            constraints += [
                x[k + 1] == self.A_d @ x[k] + self.B_d @ u[k],
                H_x @ x[k] <= h_x,
                H_u @ u[k] <= h_u,
            ]
        # Terminal state constraint
        constraints += [H_x @ x[self.horizon] <= h_x]
        constraints += [cp.quad_form(x[self.horizon]-target_state, P) <= gamma]

        # Terminal cost
        cost += (cp.quad_form((x[self.horizon] - target_state), Q))

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.SCS)

        # Return first timestep of trajectory
        return u[0].value
