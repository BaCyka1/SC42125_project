import sympy as sp
from sympy import simplify, diff
from sympy.physics.mechanics import dynamicsymbols

# --- Planar drone with 2 DOF arm ---


# Define time and dynamic symbols
t = sp.symbols('t')
x0, y0, psi0, theta1, theta2 = dynamicsymbols('x0 y0 psi0 theta1 theta2')  # functions of time

# Define system parameters: link lengths, masses, moment of inertia, and gravity
L, l, m0, m1, m2, I0, Il, g = sp.symbols('L l m0 m1 m2 I0, Il g')

# --- Kinematics ---
x1 = x0 + l * sp.sin(psi0 + theta1)
y1 = y0 - l * sp.cos(psi0 + theta1)
psi1 = psi0 + theta1

x2 = x1 + l * sp.sin(psi0 + theta1 + theta2)
y2 = y1 - l * sp.cos(psi0 + theta1 + theta2)
psi2 = psi0 + theta1 + theta2


Jv0 = sp.Matrix([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])

Jw0 = sp.Matrix([[0, 0, 1, 0, 0]])

Jv1 = sp.Matrix([
    [sp.diff(x1, x0), sp.diff(x1, y0), sp.diff(x1, psi0), sp.diff(x1, theta1), sp.diff(x1, theta2)],
    [sp.diff(y1, x0), sp.diff(y1, y0), sp.diff(y1, psi0), sp.diff(y1, theta1), sp.diff(y1, theta2)]
])

Jw1 = sp.Matrix([[sp.diff(psi1, x0), sp.diff(psi1, y0), sp.diff(psi1, psi0), sp.diff(psi1, theta1), sp.diff(psi1, theta2)]])

Jv2 = sp.Matrix([
    [sp.diff(x2, x0), sp.diff(x2, y0), sp.diff(x2, psi0), sp.diff(x2, theta1), sp.diff(x2, theta2)],
    [sp.diff(y2, x0), sp.diff(y2, y0), sp.diff(y2, psi0), sp.diff(y2, theta1), sp.diff(y2, theta2)]
])

Jw2 = sp.Matrix([[sp.diff(psi2, x0), sp.diff(psi2, y0), sp.diff(psi2, psi0), sp.diff(psi2, theta1), sp.diff(psi2, theta2)]])

print("Jacobians:")
print("Drone body")
sp.pprint(Jv0)
print()
sp.pprint(Jw0)
print("------------------------------")
print("link 1")
sp.pprint(Jv1)
print()
sp.pprint(Jw1)
print("------------------------------")
print("link 2")
sp.pprint(Jv2)
print()
sp.pprint(Jw2)
