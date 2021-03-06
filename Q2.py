# BME355 Assignment 1, Question 2

# Simulate the 3D mass/Hill-muscle system from the lecture notes.
# Implement the Explicit Trapezoid Integration method from scratch.
# Find a suitable time step and plot state vs time for 2s.

import sys
import numpy as np
import matplotlib.pyplot as plt

# Hill muscle system setup 
a, b = 1000, 1
alpha = 20000
p0 = 2000
g = 9.807
m = 220

# Define dynamic equations for 3D mass/Hill-muscle system
def f(x):
	"""
	x: state vector
	returns: derivative of x
	"""
	return np.array([
		alpha * (x[1] + b * (p0 - x[0]) / (x[0] + a)),
		g - x[0] / m,
		x[1]
	])


def expl_trap(dt):
	"""
	dt: timestep to run simulation on
	
	saves plots as .png for each timestep in dts
	"""
	time = np.arange(0, 2, dt)  # time range for 2 secconds
	x = np.array([0, 0, 0])  # initial state is x = [0, 0, 0]
	x_trajectory = []

	# implement the explicit trapezoid integration method
	for t in time:
		x_trajectory.append(x)
		xa = f(x)
		xb = f(x + dt * xa)
		x = x + 0.5 * dt * (xa + xb)
		
	# Plot the state vs time graphs for the three dynamic equations
	x_trajectory = np.array(x_trajectory)
	plt.subplot(3, 1, 1)
	plt.plot(time, x_trajectory[:, 0])
	plt.ylabel('Tension (N)')
	plt.subplot(3, 1, 2)
	plt.plot(time, x_trajectory[:, 1])
	plt.ylabel('Velocity (m/s)')
	plt.subplot(3, 1, 3)
	plt.plot(time, x_trajectory[:, 2])
	plt.ylabel('Position (m)')
	plt.tight_layout()


def find_timestep():
	"""
	saves plots as .png for each timestep in dts
	"""
	dts = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, \
		   0.015625, 0.0078125, 0.00390625, 0.001953125]

	for i in range(len(dts)):
		plt.figure(i)
		expl_trap(dts[i])
		pltname = 'q2_plots/q2_t{}.png'.format(i)
		plt.savefig(pltname)
		plt.close()


if __name__ == "__main__":
	find_timestep(*sys.argv[1:])
