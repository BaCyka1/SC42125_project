# Solution repository for SC42125 - MPC
## General information
Welcome to the repository for our project solution. It contains all the code written for the project.

The entire codebase is written in Python 3.10, so no builds are required. It requires the following packages and their dependencies:
- `numpy`
- `sympy`
- `scipy`
- `cvxpy`
- `control`
- `IPython`
- `matplotlib`

For the analysis Jupyter notebook file you will additionally need:
- `qpsolvers
- `shapely`
- `geopandas`

All packages are of their latest version as of writing (2025-04-13).

All of the latest code is on the `main` branch.
## Structure
The project is structured as follows:
- classes
	- Contains all the classes used in the project
		- MPC controller
		- drone
		- numerical simulation
	- Drone-Tutorial
		- Contains Jupyter notebook with prototyping / research code based on https://github.com/Luyao787/MPC-tutorial
	- dynamics
		- Contains symbolic dynamics derivation used as basis for model of system
	- feasibility_standalone
		- Contains all the code that checks feasibility / admissibility
	- plotting
		- Some misc code used to generate plots for the report
	- terminal_set_computation
		- Contains updated code from Drone-Tutorial. Used to compute the scaling of the ellipsoidal terminal set

## The demo
Want to demo the code? Run `main.py`. One can also easily play around the weights, constraints, and reference outputs in the same file -- just change the values in the arrays.
A video of the simulation is also available through this link: https://www.youtube.com/watch?v=dvh5GzKWKmA
