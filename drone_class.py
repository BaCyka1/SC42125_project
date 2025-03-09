import numpy as np

class Drone:
    def __init__(self, m_d=1.0, m_l=0.1, I_d=0.8, d=2.0, L_l=1.5, state_init=np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        # -- Physics --
        self.m_d = m_d  # Drone mass (kg)
        self.m_l = m_l  # Load mass (kg)
        self.d = d  # Distance between rotors (m)
        self.I_d = I_d  # Drone moment of inertia (kg*m^2)
        self.L_d = self.d / 2  # Distance from drone COM to each thruster (m)
        self.L_l = L_l  # Length of the massless rod suspending the load (m)
        self.constants = np.array([m_d, m_l, I_d, self.L_d, L_l]) # All constant values
        self.state = state_init # State: [x, y, psi, theta, vx, vy, omega, theta_dot]

        # -- Sprite --


    def update_state(self, state):
        self.state=state

