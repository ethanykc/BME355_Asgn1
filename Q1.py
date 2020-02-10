# BME355 Assignment 1, Question 1

# b) Simulate this oscillator using Euler’s method, with a time step of 0.05s, for 10s. \
# Plot the state vs. time. Write all the code from scratch using Numpy.

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# function for the derivative of state vector
def f(x):
	return np.matmul([[0, -1], [1, 0]], x)


# Implementation of Euler's method for given system
def euler(dt):
	times = np.arange(0, 10, dt)
	x_trajectory = []
	t = 0
	x = np.array([1, 0])

	for t in times:
		x_trajectory.append(x)
		t = t + dt
		x = x + f(x) * dt

	plt.plot(times, x_trajectory)
	plt.xlabel('Time (s)')
	plt.ylabel('State')
	plt.tight_layout()


# c) Implement the RK4 integration method. Simulate for 10 s with a timestep of 0.05 s
def rk4_update(dt):
	times = np.arange(0, 10, dt)
	x_trajectory = []
	t = 0
	x = np.array([1, 0])

	for t in times:
		s1 = f(x)
		s2 = f(x + 0.5 * dt * s1)
		s3 = f(x + 0.5 * dt * s2)
		s4 = f(x + dt * s3)
		x = x + dt * (s1 + (2 * s2) + (2 * s3) + s4) / 6
		x_trajectory.append(x)

	plt.plot(times, x_trajectory)
	plt.xlabel('Time (s)')
	plt.ylabel('State')
	plt.tight_layout()


# callable function for solve_ivp method
# includes t as parameter
def g(t,x):
	return np.matmul([[0, -1], [1, 0]], x)


# d) Simulate the system using solve_ivp
# Simulate for 10s with a timestep of 0.05s
def use_ivp():
	sol = solve_ivp(g, [0, 10], [1, 0], max_step=.05)
	plt.plot(sol.t, sol.y.T)
	plt.xlabel('Time (s)')
	plt.show()


def find_timestep(method):

	dts = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, \
			0.015625, 0.0078125, 0.00390625, 0.001953125]

	if method == "euler":
		for i in range(len(dts)):
			plt.figure(i)
			title_eulers = 'Simulation of Oscillator using Euler\'s method (dt = {})'.format(dts[i])
			plt.title(title_eulers)
			euler(dts[i])
			fig_name = 'q1_plots/q1e_euler_t_{}.png'.format(i)
			plt.savefig(fig_name)
			plt.close()

	elif method == "rk4":
		for i in range(len(dts)):
			plt.figure(i)
			title_rk4 = 'Simulation of Oscillator using RK-4 method (dt = {})'.format(dts[i])
			plt.title(title_rk4)
			rk4_update(dts[i])
			fig_name = 'q1_plots/q1e_rk4_t_{}.png'.format(i)
			plt.savefig(fig_name)
			plt.close()


if __name__ == "__main__":
	find_timestep(*sys.argv[1:])