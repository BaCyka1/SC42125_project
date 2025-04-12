import numpy as np
import cvxpy as cp
from control import dare
from scipy.linalg import expm
from time import time
import terminal_set_computation as tsc
from feasibility_standalone.feasibility import check_feasibility


class MPCController:
    def __init__(self, drone, constraints, horizon=20, dt=1/10, Q=np.eye(8), R=np.eye(2), x_ref=np.zeros(8)):
        print("setting up...")
        self.horizon = horizon
        self.timestep = dt
        self.drone = drone
        self.x_ref = x_ref

        # Conversion to discrete time
        dim_x_d = drone.A_c.shape[0]
        dim_u_d = drone.B_c.shape[1]

        # Generate discrete time dynamics
        AB_c = np.zeros((dim_x_d + dim_u_d, dim_x_d + dim_u_d))
        AB_c[:dim_x_d, :dim_x_d] = drone.A_c
        AB_c[:dim_x_d, dim_x_d:] = drone.B_c
        expm_AB_c = expm(AB_c * dt)

        self.A_d = expm_AB_c[:dim_x_d, :dim_x_d]
        self.B_d = expm_AB_c[:dim_x_d, dim_x_d:]

        # Unpack the constraints
        self.constraints = constraints
        self.H_x, self.H_u, self.h_x, self.h_u = constraints

        # Solve DARE for use in terminal constraints
        print("Solving Dare...")
        self.Q = Q
        self.R = R
        self.P, _, self.K = dare(self.A_d, self.B_d, Q, R)

        # Get parameter for terminal set
        print("Computing scaling factor gamma for ellipsoidal terminal set...")
        u_lb = - self.h_u[-1]
        u_ub = self.h_u[0]
        self.gamma = tsc.maximize_gamma(self.P, self.K, u_lb, u_ub)[0]
        print(f"Computed a gamma of {self.gamma}")

        # Flags
        self.admissibility_checked = False

        print("Setup complete.")

    def check_admissibility(self, current_state):
        admissible = check_feasibility(self.horizon, current_state, self.A_d, self.B_d, self.constraints, self.P, self.gamma, self.x_ref)
        if admissible:
            self.admissibility_checked = True
            print("Initial state admissible, continuing to simulation")
        else:
            print("Initial state not admissible!")
            return

    def compute_control(self, x_0):
        starttime = time()

        # Check admissibility for initial state
        if not self.admissibility_checked:
            self.check_admissibility(x_0)

        # Hovering target
        u_ref = np.array([self.drone.m * self.drone.g / 2, self.drone.m * self.drone.g / 2])

        # Decision variables
        x = [cp.Variable(8) for _ in range(self.horizon + 1)]
        u = [cp.Variable(2) for _ in range(self.horizon)]

        # Constraints and initial cost
        constraints = []
        cost = 0.

        # Initial state
        constraints += [x[0] == x_0.flatten()]

        # Stage cost and stage constraints
        for k in range(self.horizon):
            cost += (cp.quad_form((x[k+1] - self.x_ref), self.Q)  + cp.quad_form(u[k] - u_ref, self.R))
            constraints += [
                x[k + 1] == self.A_d @ x[k] + self.B_d @ u[k],
                self.H_x @ x[k] <= self.h_x,
                self.H_u @ u[k] <= self.h_u,
            ]
        # Terminal state constraint
        # constraints += [self.H_x @ x[self.horizon] <= self.h_x]
        constraints += [cp.quad_form(x[self.horizon] - self.x_ref, self.P) <= self.gamma] # Optional terminal set

        # Terminal cost - Unconstrained infinite-horizon optimal cost
        cost += (cp.quad_form((x[self.horizon] - self.x_ref), self.P))

        # Solve the problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.CLARABEL) # cp.SCS

        print(f"Time spent on this step's optimization: {time() - starttime}")
        # Return first timestep of input sequence
        return u[0].value
