import numpy as np

class Drone:
    def __init__(self, m_d=1.0, m_l=0.1, I_d=0.8, d=2.0, L_l=1.5, g=9.81, x_0=np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        # -- Physics --
        self.m_d = m_d  # Drone mass (kg)
        self.m_l = m_l  # Load mass (kg)
        self.m = m_d + m_l
        self.d = d  # Distance between rotors (m)
        self.I_d = I_d  # Drone moment of inertia (kg*m^2)
        self.L_d = self.d / 2  # Distance from drone COM to each thruster (m)
        self.L_l = L_l  # Length of the massless rod suspending the load (m)
        self.g = g
        self.constants = np.array([m_d, m_l, I_d, self.L_d, L_l]) # All constant values
        self.state = x_0 # State: [x, y, psi, theta, vx, vy, psi_dot, theta_dot]

        # Dynamics
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
        self.B_c[6, 0] = - self.L_d / I_d
        self.B_c[6, 1] = self.L_d / I_d


    def update_state(self, state):
        self.state=state
