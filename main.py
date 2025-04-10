import numpy as np
import simulation_class as sim
from drone_class import Drone
from MPC_class import MPCController

# Linear state and input constraints
H_x = np.vstack((np.eye(8), np.eye(8) * -1))

h_x = np.ones(H_x.shape[0]) * 100

H_u = np.array([[1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]])

h_u = np.ones(H_u.shape[0]) * 10

constraints = (H_x, H_u, h_x, h_u)

# State and input penalties
Q = np.diag([10, 10, 5, 10, 1, 1, 1, 5])  # State penalties
R = np.eye(2) * 0.01  # Control input penalties

# Instantiate classes
drone = Drone()
controller = MPCController(drone, horizon=20, dt=1/8, constraints=constraints, Q=Q, R=R)
simulation = sim.Simulation(drone, controller)

# Run the simulation
simulation.run_simulation()
