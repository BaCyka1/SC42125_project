import numpy as np
import cvxpy as cp
from scipy.linalg import expm
from time import time


def gen_prediction_matrices(Ad, Bd, N):
    dim_x = Ad.shape[0]
    dim_u = Bd.shape[1]

    T = np.zeros(((dim_x * (N + 1), dim_x)))
    S = np.zeros(((dim_x * (N + 1), dim_u * N)))

    # Condensing
    power_matricies = []  # power_matricies = [I, A, A^2, ..., A^N]
    power_matricies.append(np.eye(dim_x))
    for k in range(N):
        power_matricies.append(power_matricies[k] @ Ad)

    for k in range(N + 1):
        T[k * dim_x: (k + 1) * dim_x, :] = power_matricies[k]
        for j in range(N):
            if k > j:
                S[k * dim_x:(k + 1) * dim_x, j * dim_u:(j + 1) * dim_u] = power_matricies[k - j - 1] @ Bd

    return T, S


dim_x_d = 8
dim_u_d = 2

m_d = 1.0  # Drone mass (kg)
m_l = 0.1  # Load mass (kg)
g = 9.81  # Gravity (m/s^2)
d = 2.0  # Distance between rotors (m)
I_d = 0.8  # Drone moment of inertiA_d (kg*m^2)
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
x0 = np.array([3, 2, np.pi/4, 0, 3, 0, 0, 0])
x0[0] = 3
x0[1] = 2
x0[2] = np.pi/4
x0[4] = 3

# Horizon (modifiable)
N = 100  # <-- Change this value as needed

T, S = gen_prediction_matrices(A_d, B_d, N)

u = cp.Variable((dim_u_d * N))
x_stack = T @ x0 + S @ u

# State and input constraints
H_x = np.random.randn(4, dim_x_d)
h_x = np.ones(4)

H_u = np.random.randn(2, dim_u_d)
h_u = np.ones(2) * 1

constraints = []

# State constraints for each timestep
for k in range(N):
    xk = x_stack[k*dim_x_d:(k+1)*dim_x_d]
    constraints += [H_x @ xk <= h_x]

# Terminal state constraint
xN = x_stack[N*dim_x_d:(N+1)*dim_x_d]
constraints += [H_x @ xN <= h_x]

# Input constraints
for k in range(N):
    uk = u[k*dim_u_d:(k+1)*dim_u_d]
    constraints += [H_u @ uk <= h_u]

starttime = time()
# Problem setup: no objective, just feasibility
prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve()

print(f"Time to solve feasibility LP: {time()-starttime}")

# Result
if prob.status in ["optimal", "optimal_inaccurate"]:
    print(f"Initial state x0 is feasible for horizon N = {N}.")
else:
    print(f"Initial state x0 is NOT feasible for horizon N = {N}.")
