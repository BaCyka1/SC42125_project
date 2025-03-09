import numpy as np
import simulation_class as sim
from drone_class import Drone
from MPC_class import MPCController

drone = Drone()
controller = MPCController(drone, horizon=20, controller_timestep=1/10)
simulation = sim.Simulation(drone, controller)

simulation.run_simulation()