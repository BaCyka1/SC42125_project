import numpy as np
import simulation_class as sim
from drone_class import Drone
from MPC_class import MPCController

# Linear state and input constraints
H_x = np.vstack((np.eye(8), np.eye(8) * -1))

h_x = np.ones(H_x.shape[0]) * 10

H_u = np.array([[1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]])

h_u = np.ones(H_u.shape[0]) * 10

constraints = (H_x, H_u, h_x, h_u)

# State and input penalties
Q = np.diag([10, 10, 5, 10, 1, 1, 1, 10])  # State penalties
R = np.eye(2) * 1  # Control input penalties

# Target state
x_ref = np.array([7.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Instantiate classes
drone = Drone()
controller = MPCController(drone, horizon=50, dt=1/10, constraints=constraints, Q=Q, R=R, x_ref=x_ref)
simulation = sim.Simulation(drone, controller)

# Run the simulation
simulation.run_simulation(frames=6000)
