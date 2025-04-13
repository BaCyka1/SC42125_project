import numpy as np
import itertools
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize


def get_ellipsoid_polytope_corners(P, gamma):
    """
    Construct polytope corners from ellipsoid defined by x.T P x <= gamma
    using aligned box approximation (principal axes of P).
    """
    eigvals, eigvecs = np.linalg.eigh(P)
    axes_lengths = np.sqrt(gamma / eigvals)  # semi-axis lengths

    dim = P.shape[0]
    signs = np.array(list(itertools.product([-1, 1], repeat=dim)))  # 2^n sign combinations
    A = np.diag(axes_lengths)
    corners = (eigvecs @ A @ signs.T).T  # Shape: (2^dim, dim)
    return corners


def check_input_constraints_on_corners(K, corners, u_lb, u_ub, H_x, h_x):
    """
    Check if all control inputs u = Kx (for each corner x) are within bounds.
    """
    all_feasible = True
    for i, x_corner in enumerate(corners):
        u = K @ x_corner
        if not np.all((u_lb <= u) & (u <= u_ub)):
            if not np.all(H_x @ x_corner <= h_x):
                # print(f"[N] Corner {i}: u = {np.round(u,3)} violates bounds.")
                all_feasible = False
        else:
            # print(f"[Y] Corner {i}: u = {np.round(u,3)} within bounds.")
            pass
    return all_feasible


def maximize_gamma(P, K, u_lb, u_ub, H_x, h_x):
    # Objective function: Maximize gamma
    def objective(gamma):
        corners = get_ellipsoid_polytope_corners(P, gamma)
        if check_input_constraints_on_corners(K, corners, u_lb, u_ub, H_x, h_x):
            return -gamma  # Minimize negative gamma to maximize gamma
        else:
            return 1e6  # Penalize infeasible solutions

    # Initial guess for gamma
    gamma_initial = 0
    # Constraints: gamma must be positive
    bounds = [(0, 1000)]

    # Run optimization using scipy
    result = scipy.optimize.minimize(objective, gamma_initial, bounds=bounds, method='SLSQP')
    gamma = result.x

    return gamma

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
