import numpy as np
import simulation_class as sim
from drone_class import Drone
from MPC_class import MPCController
import matplotlib.pyplot as plt

# Linear state and input constraints - box constraints
H_x = np.vstack((np.eye(8), np.eye(8) * -1))

h_x = np.ones(H_x.shape[0]) * 100

# State constraint: Do not deviate far from equilibrium
# to preserve validity of approximation
# Absolute difference between theta and psi < pi/6
h_x[2] = np.pi/2
h_x[6] = np.pi/2
h_x[3] = np.pi/2
h_x[7] = np.pi/2
H_x[3, 2] = -1

# Input constraints - lower and upper bound on thrust
H_u = np.array([[1, 0],
                [0, 1],
                [-1, 0],
                [0, -1]])

h_u = np.ones(H_u.shape[0]) * 20

constraints = (H_x, H_u, h_x, h_u)

# State and input penalties, tuned to stay near equilibrium
Q = np.diag([1, 20, 20, 20, 1, 1, 5, 5])  # State penalties
R = np.eye(2) * 0.01  # Control input penalties

# Initial state
x_0 = np.array([0.0, 5.0, 0, 0, 0.0, 0.0, 0.0, 0.0])

# Reference output
y_ref = np.array([8, 8, 0])

drone = Drone(x_0=x_0)
controller = MPCController(drone, horizon=40, dt=1/10, constraints=constraints, Q=Q, R=R, y_ref=y_ref)
simulation = sim.Simulation(drone, controller, dt=1/100, linear=False)
simulation.run_simulation(frames=600)
run1_states = np.array(simulation.state_history)

drone = Drone(x_0=x_0)
controller = MPCController(drone, horizon=10, dt=1/10, constraints=constraints, Q=Q, R=R, y_ref=y_ref)
simulation = sim.Simulation(drone, controller, dt=1/100, linear=False, LQR=True)
simulation.run_simulation(frames=600)
run2_states = np.array(simulation.state_history)


labels = ["MPC", "LQR"]

def plot_column_from_arrays(arrays, column_index, dt, labels=None):
    """
    Plots a specified column from multiple nx8 numpy arrays against time on the same figure.
    """

    plt.figure(figsize=(8, 6))  # Create the figure only once

    for i, array in enumerate(arrays):
        if not isinstance(array, np.ndarray):
             raise TypeError(f"Array {i+1} in 'arrays' list is not a NumPy array.")
        if array.ndim != 2:  # Check that the array is 2D
            raise ValueError(f"Array {i+1} is not a 2D array.")
        if array.shape[1] != 8:
            raise ValueError(f"Array {i+1} does not have 8 columns. Has {array.shape[1]} instead.")
        n_rows = array.shape[0]
        time = np.arange(n_rows) * dt

        column_values = array[:, column_index]

        if labels:
            plt.plot(time, column_values, label=labels[i])
        else:
            plt.plot(time, column_values)

    plt.xlabel("Time")
    plt.ylabel(f"Value in Column {column_index}")
    plt.title(f"Column {column_index} vs. Time for Multiple Arrays")
    plt.grid(True)

    if labels:
        plt.legend()

    plt.show()

plot_column_from_arrays([run1_states, run2_states], 1, 0.01, labels=labels)