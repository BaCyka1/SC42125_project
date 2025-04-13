import numpy as np
import classes.simulation_class as sim
from classes.drone_class import Drone
from classes.MPC_class import MPCController

# Linear state and input constraints - box constraints
H_x = np.vstack((np.eye(8), np.eye(8) * -1))

h_x = np.ones(H_x.shape[0]) * 10

# State constraint: Do not deviate far from equilibrium to preserve validity of approximation
# Absolute difference between theta and psi < pi/6
h_x[2] = np.pi/8
h_x[3] = np.pi/8
H_x[3, 2] = -1

# Input constraints - lower and upper bound on thrust
H_u = np.array([[1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]])

h_u = np.ones(H_u.shape[0]) * 20

constraints = (H_x, H_u, h_x, h_u)

# State and input penalties, tuned to stay near equilibrium
Q = np.diag([1, 1, 20, 20, 1, 1, 5, 5])  # State penalties
R = np.eye(2) * 0.1  # Control input penalties

# Initial state
x_0 = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Reference output
y_ref = np.array([6, 9, 0])

# Instantiate classes
drone = Drone(x_0=x_0)
controller = MPCController(drone, horizon=40, dt=1/10, constraints=constraints, Q=Q, R=R, y_ref=y_ref)
simulation = sim.Simulation(drone, controller, dt=1/100, linear=False, LQR=False)

# Run the simulation
simulation.run_simulation(frames=600)
