import sympy as sp
from sympy import simplify, diff
from sympy.physics.mechanics import dynamicsymbols

# Define time and dynamic symbols for joint angles
t = sp.symbols('t')
q1, q2 = dynamicsymbols('q1 q2')  # functions of time: q1(t), q2(t)
q1_dot = diff(q1, t)
q2_dot = diff(q2, t)

# Define system parameters: link lengths, masses, moments of inertia, and gravity
l1, l2, m1, m2, I1, I2, g = sp.symbols('l1 l2 m1 m2 I1 I2 g')

# --- Kinematics: positions of centers of mass ---

# For link 1, assume the center of mass is located at end of the link
x1 = l1 * sp.cos(q1)
y1 = l1 * sp.sin(q1)

# For link 2, first compute the position of the second joint (end of link 1)
x_joint2 = l1 * sp.cos(q1)
y_joint2 = l1 * sp.sin(q1)

# Then, the center of mass of link 2 is at the end of link 2
x2 = x_joint2 + l2 * sp.cos(q1 + q2)
y2 = y_joint2 + l2 * sp.sin(q1 + q2)

# --- Velocities: time derivatives of the positions ---
x1_dot = diff(x1, t)
y1_dot = diff(y1, t)
x2_dot = diff(x2, t)
y2_dot = diff(y2, t)

# --- Kinetic Energy (T) ---
# Translational kinetic energy for each link: (1/2)*m*v^2
T1 = 0.5 * m1 * (x1_dot**2 + y1_dot**2)
T2 = 0.5 * m2 * (x2_dot**2 + y2_dot**2)

T = simplify(T1 + T2)

# --- Potential Energy (V) ---
# Assume that the potential energy is due to gravity (using y-coordinate)
V1 = m1 * g * y1
V2 = m2 * g * y2

V = V1 + V2

# --- Lagrangian (L) ---
L = T - V

# --- Euler-Lagrange Equations ---
# For each generalized coordinate q_i, the Euler-Lagrange equation is:
#   d/dt(∂L/∂q̇_i) - ∂L/∂q_i = 0

# For q1:
EL_eq1 = simplify(diff(diff(L, q1_dot), t) - diff(L, q1))
# For q2:
EL_eq2 = simplify(diff(diff(L, q2_dot), t) - diff(L, q2))

# Display the equations of motion
print("Euler-Lagrange Equation for q1:")
sp.pprint(EL_eq1)

print("\nEuler-Lagrange Equation for q2:")
sp.pprint(EL_eq2)

print("Euler-Lagrange Equation for q1 (LaTeX format):")
print(sp.latex(EL_eq1))

print("\nEuler-Lagrange Equation for q2 (LaTeX format):")
print(sp.latex(EL_eq2))
