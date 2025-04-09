import numpy as np
import cvxpy as cp
from scipy.linalg import expm
from time import time


def check_feasibility(N, x0, A_d, B_d, problem_constraints):
    """
    Checks feasibility for a single initial state by attempting to solve an LP
    """
    # Get shapes for dynamics
    dim_x_d = A_d.shape[0]
    dim_u_d = B_d.shape[1]

    # Unpack the constraints
    H_x, H_u, h_x, h_u = problem_constraints

    # Variables
    x = [cp.Variable(dim_x_d) for _ in range(N + 1)]
    u = [cp.Variable(dim_u_d) for _ in range(N)]

    # Constraints
    constraints = [x[0] == x0]
    for k in range(N):
        constraints += [
            x[k + 1] == A_d @ x[k] + B_d @ u[k],
            H_x @ x[k] <= h_x,
            H_u @ u[k] <= h_u
        ]
    # Terminal state constraint
    constraints += [H_x @ x[N] <= h_x]

    starttime = time()
    # Problem setup: no objective, just feasibility
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    print(f"Time to solve feasibility LP: {time() - starttime}")

    # Result
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return True
    else:
        return False


def compute_bounding_box_cvxpy(H_x, h_x, default_bound=10.0):
    """
    Finds the upper and lower bounds of a bounding box around the state constraints.
    This box is used for sampling states during the feasibility check.
    """
    dim = H_x.shape[1]
    bounds = []
    x = cp.Variable(dim)

    constraints = [H_x @ x <= h_x]

    for i in range(dim):
        # Lower bound
        obj_min = cp.Minimize(x[i])
        prob_min = cp.Problem(obj_min, constraints)
        prob_min.solve()

        if prob_min.status == cp.OPTIMAL:
            lower = prob_min.value
        elif prob_min.status == cp.UNBOUNDED:
            lower = -default_bound

        # Upper bound
        obj_max = cp.Maximize(x[i])
        prob_max = cp.Problem(obj_max, constraints)
        prob_max.solve()

        if prob_max.status == cp.OPTIMAL:
            upper = prob_max.value
        elif prob_max.status == cp.UNBOUNDED:
            upper = default_bound

        bounds.append((lower, upper))

    return np.array(bounds)


def collect_feasible_samples(n_samples, N, A_d, B_d, constraints):
    H_x, _, h_x, _ = constraints
    bounds = compute_bounding_box_cvxpy(H_x, h_x)

    feasible_samples = []
    while len(feasible_samples) < n_samples:

        x0 = np.array([np.random.uniform(low, high) for (low, high) in bounds])

        if np.all(H_x @ x0 <= h_x):
            result = check_feasibility(N, x0, A_d, B_d, constraints)
            if result:
                feasible_samples.append(x0)
    feasible_samples = np.array(feasible_samples)
    return feasible_samples


def mvee(points, tol=1e-5):
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T  # (d+1) x N

    P = cp.Variable(N)
    objective = cp.Maximize(cp.log_det(Q @ cp.diag(P) @ Q.T))
    constraints = [P >= 0, cp.sum(P) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Compute the ellipsoid parameters
    P_val = P.value
    center = points.T @ P_val
    cov_matrix = (points - center).T @ np.diag(P_val) @ (points - center) / d
    A = np.linalg.inv(cov_matrix)

    return center, A


def estimate_feasible_region(n_samples, N, A_d, B_d, constraints):
    feasible_samples = collect_feasible_samples(n_samples, N, A_d, B_d, constraints)
    center, A = mvee(feasible_samples)
    return center, A


dim_x_d = 8
dim_u_d = 2

m_d = 1.0  # Drone mass (kg)
m_l = 0.1  # Load mass (kg)
g = 9.81  # Gravity (m/s^2)
d = 2.0  # Distance between rotors (m)
I_d = 0.8  # Drone moment of inertia_d (kg*m^2)
L_d = d / 2  # Distance from drone COM to each thruster (m)
L_l = 1.5  # Length of the massless rod suspending the load (m)

A_c = np.zeros(shape=(8,8))
A_c[:4, 4:] = np.eye(4)
A_c[4, 2] = - g * (m_d + m_l) / m_d
A_c[4, 3] = g * m_l / m_d
A_c[7, 2] = g * (m_d + m_l) / (m_d * L_l)
A_c[7, 3] = - g * (m_d + m_l) / (m_d * L_l)

B_c = np.zeros(shape=(8, 2))
B_c[5, 0] = 1/(m_d + m_l)
B_c[5, 1] = 1 / (m_d + m_l)
B_c[6, 0] = - L_d / I_d
B_c[6, 1] = L_d / I_d

dt = 0.1
# Trick to compute the matrix exponential
AB_c = np.zeros((dim_x_d + dim_u_d, dim_x_d + dim_u_d))
AB_c[:dim_x_d, :dim_x_d] = A_c
AB_c[:dim_x_d, dim_x_d:] = B_c
expm_AB_c = expm(AB_c*dt)
A_d = expm_AB_c[:dim_x_d, :dim_x_d]
B_d = expm_AB_c[:dim_x_d, dim_x_d:]

# Initial state
x0 = np.array([1, 1, np.pi/4, 0, 1, 0, 0, 0])
x0[0] = 3
x0[1] = 2
x0[2] = np.pi/4
x0[4] = 3

# State and input constraints
# H_x = np.random.randn(4, dim_x_d)
# h_x = np.ones(4)
#
# H_u = np.random.randn(2, dim_u_d)
# h_u = np.ones(2) * 1

H_x = np.zeros((4, dim_x_d))
h_x = np.zeros(4)

H_u = np.zeros((2, dim_u_d))
h_u = np.zeros(2) * 1

N = 100
constraints = (H_x, H_u, h_x, h_u)

n_samples = 100
c, A = estimate_feasible_region(n_samples, N, A_d, B_d, constraints)
print(c)
print(A)