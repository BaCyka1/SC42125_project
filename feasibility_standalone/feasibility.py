import numpy as np
import cvxpy as cp
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from control import dare
from time import time
from scipy.spatial import ConvexHull

def check_feasibility(N, x0, A_d, B_d, problem_constraints, P, gamma, x_ref=np.zeros(8)):
    """
    Checks feasibility for a single initial state by attempting to solve a QCLP
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
            H_u @ u[k] <= h_u,
        ]
    # Terminal state constraint
    constraints += [H_x @ x[N] <= h_x]
    constraints += [cp.quad_form(x[N] - x_ref, P) <= gamma]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS)

    # Result
    if prob.status in ["optimal", "optimal_inaccurate"]:
        return True
    else:
        return False


def compute_sample_space(H_x, h_x, default_bound=100.0):
    """
    Finds the upper and lower bounds of a bounding box around the state constraints.
    This box is used for sampling states during the feasibility check.
    """
    print("Computing bounding box for sample space.")

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

    bounds = np.array(bounds)
    bounds[2, :] = [-np.pi/1, np.pi/1]
    bounds[3, :] = [-np.pi / 1, np.pi / 1]
    bounds[4:, 0] = np.ones(4) * -1
    bounds[4:, 1] = np.ones(4)
    print(bounds)
    return bounds


def collect_feasible_samples(n_samples, N, A_d, B_d, constraints):
    print("Started collection of feasible samples.")
    H_x, _, h_x, _ = constraints
    bounds = compute_sample_space(H_x, h_x)

    feasible_samples = []
    while len(feasible_samples) < n_samples:

        x0 = np.array([np.random.uniform(low, high) for (low, high) in bounds])

        if np.all(H_x @ x0 <= h_x):
            result = check_feasibility(N, x0, A_d, B_d, constraints)
            if result:
                feasible_samples.append(x0)
        if len(feasible_samples) % 20 == 0 and len(feasible_samples) != 0:
            print(f"Collecting, found {len(feasible_samples)} samples so far.")
    feasible_samples = np.array(feasible_samples)
    return feasible_samples


# Ellipsoid fitting from https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953
def mvee(points, tol=0.00001):
    print("Fitting ellipsoid to samples")
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = np.linalg.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return c, A


def project_ellipsoid_to_2d(A, center, dims):
    """
    Project an ellipsoid defined by (x - center)^T A (x - center) = 1
    onto a 2D plane defined by state index (e.g. [0, 1]).
    """
    # Extract submatrix of A and center
    A_2d = A[np.ix_(dims, dims)]
    center_2d = center[dims]

    return center_2d, A_2d

# Plotting code by GPT-4o
def plot_ellipsoid_2d(center_2d, A_2d, ax=None, edgecolor='r', facecolor='none', linewidth=2, label=None):
    """
    Plots a 2D ellipsoid defined by (x - center)^T A (x - center) = 1.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Eigen-decomposition for orientation and axes lengths
    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(A_2d))

    # Width and height should be scaled by sqrt of eigenvalues (multiplied by 2 for full axis lengths)
    width, height = 2 * np.sqrt(eigvals)  # Scaling axes length by sqrt of eigenvalues

    # Orientation of the ellipse
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # Orientation in degrees

    ellipse = Ellipse(xy=center_2d, width=width, height=height,
                      angle=angle, edgecolor=edgecolor,
                      facecolor=facecolor, linewidth=linewidth, label=label)

    ax.add_patch(ellipse)
    ax.set_aspect('equal')
    ax.autoscale_view()


def estimate_feasible_region(n_samples, N, A_d, B_d, constraints):
    feasible_samples = collect_feasible_samples(n_samples, N, A_d, B_d, constraints)
    hull = ConvexHull(feasible_samples )
    return feasible_samples, hull


def generate_dynamics():
    dim_x_d = 8
    dim_u_d = 2

    m_d = 1.0  # Drone mass (kg)
    m_l = 0.1  # Load mass (kg)
    g = 9.81  # Gravity (m/s^2)
    d = 2.0  # Distance between rotors (m)
    I_d = 0.8  # Drone moment of inertia_d (kg*m^2)
    L_d = d / 2  # Distance from drone COM to each thruster (m)
    L_l = 1.5  # Length of the massless rod suspending the load (m)

    A_c = np.zeros(shape=(8, 8))
    A_c[:4, 4:] = np.eye(4)
    A_c[4, 2] = - g * (m_d + m_l) / m_d
    A_c[4, 3] = g * m_l / m_d
    A_c[7, 2] = g * (m_d + m_l) / (m_d * L_l)
    A_c[7, 3] = - g * (m_d + m_l) / (m_d * L_l)

    B_c = np.zeros(shape=(8, 2))
    B_c[5, 0] = 1 / (m_d + m_l)
    B_c[5, 1] = 1 / (m_d + m_l)
    B_c[6, 0] = - L_d / I_d
    B_c[6, 1] = L_d / I_d

    dt = 0.1
    # Trick to compute the matrix exponential
    AB_c = np.zeros((dim_x_d + dim_u_d, dim_x_d + dim_u_d))
    AB_c[:dim_x_d, :dim_x_d] = A_c
    AB_c[:dim_x_d, dim_x_d:] = B_c
    expm_AB_c = expm(AB_c * dt)
    A_d = expm_AB_c[:dim_x_d, :dim_x_d]
    B_d = expm_AB_c[:dim_x_d, dim_x_d:]

    return A_d, B_d

# dim_x_d = 8
# dim_u_d = 2
#
# # Constraints
# H_x = np.zeros((4, dim_x_d))
# h_x = np.zeros(4)
#
# H_u = np.zeros((2, dim_u_d))
# h_u = np.zeros(2) * 1
#
#
# H_x = np.array([[1, 0, 0, 0, 0, 0, 0 ,0],
#                 [0, 1, 0, 0, 0, 0, 0 ,0],
#                 [-1, 0, 0, 0, 0, 0, 0 ,0],
#                 [0, -1, 0, 0, 0, 0, 0 ,0]])
#
#
# H_x = np.vstack((np.eye(8), np.eye(8) * -1))
#
#
# h_x = np.ones(H_x.shape[0]) * 2
#
# H_u = np.array([[1, 0],
#                 [0, 1],
#                 [-1, 0],
#                 [0, -1]])
#
# h_u = np.ones(H_u.shape[0]) * 100
#
# constraints = (H_x, H_u, h_x, h_u)
#
# A_d, B_d = generate_dynamics()
#
# N = 20
# n_samples = 100
#
# starttime = time()
# feasible_samples = collect_feasible_samples(n_samples, N, A_d, B_d, constraints)
# print(f"Total time to get all data for feasible region visualization: {time() - starttime}")
#
# Q = np.eye(8)  # State cost matrix
# R = np.eye(2)  # Input cost matrix
# P, _, _ = dare(A_d, B_d, Q, R)
#
#
# for i in range(8):
#     for j in range(8):
#         if i != j:
#             # Plot the admissible set
#             feasible_samples_2d = feasible_samples[:, [i, j]]
#             hull = ConvexHull(feasible_samples_2d)
#             hull_points = feasible_samples_2d[hull.vertices]
#             fig, ax = plt.subplots()
#             ax.fill(hull_points[:, 0], hull_points[:, 1], color='lightblue', alpha=0.5, edgecolor='blue')
#
#             # plot the terminal set
#             center = np.zeros(8)
#             center_2d, P_2d = project_ellipsoid_to_2d(P, center, [i, j])
#
#             eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(P_2d))
#
#             # Width and height
#             width, height = 2 * np.sqrt(eigvals)  # Scaling axes length by sqrt of eigenvalues
#
#             # Orientation of the ellipse
#             angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))  # Orientation in degrees
#
#             ellipse = Ellipse(center_2d, width, height, angle=angle, color='green', alpha=0.4)
#
#             ax.add_patch(ellipse)
#
#             #add origin
#             plt.plot(0, 0, 'k+', markersize=20, markeredgewidth=1)
#
#             plt.axis('equal')
#             plt.show()





# for i in range(8):
#     for j in range(8):
#         if i != j:
#             center_2d, A_2d = project_ellipsoid_to_2d(A, center, [i, j])
#             ax = plot_ellipsoid_2d(center_2d, A_2d)
#             plt.scatter(feasible_samples[:, i], feasible_samples[:, j], s=10)
#             plt.title(f"2D MVEE Projection - states {i} and {j}")
#             plt.show()


# Q = np.eye(8)  # State cost matrix
# R = np.eye(2)  # Input cost matrix
# P, _, K = dare(A_d, B_d, Q, R)
# print(P.shape)
# x0 = np.ones(8) * 2
# x0 = np.zeros(8)
# x0[:2] = [10, 10]
# res = check_feasibility(N, x0, A_d, B_d, constraints)
# print(res)


