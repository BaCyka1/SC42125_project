import sympy as sp
from sympy import simplify, diff, symbols, Matrix, solve
from sympy.physics.mechanics import dynamicsymbols

# Define time and dynamic symbols for joint angles
t = sp.symbols('t')
xd, yd, psi, theta = dynamicsymbols('x_d y_d psi theta')  # functions of time: q1, q2, q3, q4

# Define system parameters: link lengths, masses, moments of inertia, and gravity
ld, lr, md, ml, Id, g = sp.symbols('l_d l_r m_d m_l I_d g')
F1, F2 = sp.symbols('F1 F2')  # rotor thrust forces

# --- Kinematics: positions of centers of mass ---

# Location of load COM
xl = xd + lr * sp.sin(theta)
yl = yd - lr * sp.cos(theta)

# --- Velocities: time derivatives ---
xd_dot = diff(xd, t)
yd_dot = diff(yd, t)
psi_dot = diff(psi, t)
theta_dot = diff(theta, t)
xl_dot = diff(xl, t)
yl_dot = diff(yl, t)

# --- Kinetic Energy (T) ---
# Translational and rotational kinetic energy for drone and load
Td = 0.5 * md * (xd_dot**2 + yd_dot**2) + 0.5 * Id * psi_dot**2
Tl = 0.5 * ml * (xl_dot**2 + yl_dot**2)

T = simplify(Td + Tl)

# --- Potential Energy (V) ---
# Potential energy due to gravity
Vd = md * g * yd
Vl = ml * g * yl

V = Vd + Vl

# --- Lagrangian (L) ---
L = T - V

# --- Generalized Control Forces ---
F_total = F1 + F2
# Map rotor thrusts into the inertial frame (assuming psi=0 means thrust in +y)
Q_xd = -F_total * sp.sin(psi)
Q_yd = F_total * sp.cos(psi)
Q_psi = ld * (F2 - F1)
Q_theta = 0  # No direct control on theta

# --- Euler-Lagrange Equations with Control Inputs ---
EL_eq1 = simplify(diff(diff(L, xd_dot), t) - diff(L, xd) - Q_xd)
EL_eq2 = simplify(diff(diff(L, yd_dot), t) - diff(L, yd) - Q_yd)
EL_eq3 = simplify(diff(diff(L, psi_dot), t) - diff(L, psi) - Q_psi)
EL_eq4 = simplify(diff(diff(L, theta_dot), t) - diff(L, theta) - Q_theta)

# Display the equations of motion
# print("Euler-Lagrange Equation for q1:")
# sp.pprint(EL_eq1)
#
# print("\nEuler-Lagrange Equation for q2:")
# sp.pprint(EL_eq2)
#
# print("Euler-Lagrange Equation for q3:")
# sp.pprint(EL_eq3)
#
# print("\nEuler-Lagrange Equation for q4:")
# sp.pprint(EL_eq4)

print("Euler-Lagrange Equation for q1 (LaTeX format):")
print(sp.latex(EL_eq1))

print("\nEuler-Lagrange Equation for q2 (LaTeX format):")
print(sp.latex(EL_eq2))

print("\nEuler-Lagrange Equation for q3 (LaTeX format):")
print(sp.latex(EL_eq3))

print("\nEuler-Lagrange Equation for q4 (LaTeX format):")
print(sp.latex(EL_eq4))

xd_ddot = diff(xd_dot, t)
yd_ddot = diff(yd_dot, t)
psi_ddot = diff(psi_dot, t)
theta_ddot = diff(theta_dot, t)

solution = solve([EL_eq1, EL_eq2, EL_eq3, EL_eq4], [xd_ddot, yd_ddot, psi_ddot, theta_ddot], dict=True)

if solution:  # Ensure the solution is not empty
    solution = solution[0]  # Extract the first dictionary from the list

    print("Solutions in LaTeX format:")
    for var, sol in solution.items():
        print(f"{var} =", sp.latex(sol))  # Convert each solution to LaTeX format
else:
    print("No symbolic solution found.")


# --- Linearization of the System ---

# Define new symbols to represent the state variables (independent symbols)
x1, x2, x3, x4, x5, x6, x7, x8 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8')

# Create a substitution dictionary mapping the original dynamic symbols to state variables:
subs_dict = {
    xd: x1,       # Drone x-position
    yd: x2,       # Drone y-position
    psi: x3,      # Drone orientation
    theta: x4,    # Load cable angle
    xd_dot: x5,   # Drone x-velocity
    yd_dot: x6,   # Drone y-velocity
    psi_dot: x7,  # Drone angular rate
    theta_dot: x8 # Load angular rate
}

# Construct the state derivative vector f(x,u)
f1 = x5
f2 = x6
f3 = x7
f4 = x8
f5 = solution[xd_ddot]
f6 = solution[yd_ddot]
f7 = solution[psi_ddot]
f8 = solution[theta_ddot]
f = Matrix([f1, f2, f3, f4, f5, f6, f7, f8])

# Substitute the dynamic symbols with state variables in f
f_state = f.subs(subs_dict)

# --- Small Angle Approximations ---
# Replace sin(·) and cos(·) with their small-angle equivalents
# (Note: Here, psi and theta in the small_angle_subs remain as the state symbols x3 and x4 after substitution)
small_angle_subs = {
    sp.sin(x3): x3,
    sp.cos(x3): 1,
    sp.sin(x4): x4,
    sp.cos(x4): 1,
    sp.sin(x3 + x4): x3 + x4,
    sp.cos(x3 + x4): 1
}

# Apply the small angle approximations to the state dynamics
f_state_small = f_state.subs(small_angle_subs)

# Define the state vector (now in terms of the independent symbols)
state_vector = Matrix([x1, x2, x3, x4, x5, x6, x7, x8])

# Compute the Jacobians with respect to the state and inputs
A = sp.simplify(f_state_small.jacobian(state_vector))
B = sp.simplify(f_state_small.jacobian(Matrix([F1, F2])))

# --- Equilibrium Substitution ---
# At equilibrium (e.g., hover) we assume the angles and angular rates are zero.
# Note: Since our state variables for angles and angular rates are x3, x4, x7, and x8,
# we substitute these with zero. We also set F1 and F2 to their equilibrium values.
# equilibrium_subs = {x3: 0, x4: 0, x7: 0, x8: 0,
#                     F1: 0.5*(md + ml)*g, F2: 0.5*(md + ml)*g}

equilibrium_subs = {x3: 0, x4: 0, x7: 0, x8: 0, F1: 0.5 * g * (md + ml), F2: 0.5 * g * (md + ml)}

A_eq = sp.simplify(A.subs(equilibrium_subs))
B_eq = sp.simplify(B.subs(equilibrium_subs))

print("\nLinearized A matrix with small angle approximations at equilibrium:")
print(sp.latex(A_eq))
print("\nLinearized B matrix with small angle approximations at equilibrium:")
print(sp.latex(B_eq))