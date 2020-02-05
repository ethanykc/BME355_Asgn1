# BME355 Assignment 1, Question 1

# b) Simulate this oscillator using Euler’s method, with a time step of 0.05s, for 10s. \
# Plot the state vs. time. Write all the code from scratch using Numpy.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# function for the derivative of state vector
def f(x):
	return np.matmul([[0, -1], [1, 0]], x)


# Implementation of Euler's method for given system
def euler(dt):
	times = []
	trajectory = []
	t = 0
	x = np.array([1, 1])

	for i in range(int(10/dt)):
		times.append(t)
		trajectory.append(x)
		t = t + dt
		x = x + f(x)*dt

	plt.plot(times, trajectory)	

	plt.xlabel('Time (s)')
	plt.ylabel('State')
	plt.tight_layout()
	plt.show()


def rk4_update(dt):
	times = np.arange(0,10,dt)
	x_trajectory = []
	t = 0
	x = np.array([1, 0])

	for t in times:  
		s1 = x + dt * f(x)
		s2 = s1 + 0.5 * dt * f(s1)
		s3 = s2 + 0.5 * dt * f(s2)
		s4 = s3 + dt * f(s3)
		x = x + dt * (s1 + (2*s2) + (2*s3) + s4)/6
		x_trajectory.append(x)

	plt.plot(times, x_trajectory)	

	plt.xlabel('Time (s)')
	plt.ylabel('State')
	plt.tight_layout()
	plt.show()


# callable function for solve_ivp method 
# includes t as parameter
def g(t,x):
	return np.matmul([[0, -1], [1, 0]], x)

def use_ivp():
	sol = solve_ivp(g, [0, 10], [1, 0], max_step=.05)
	plt.plot(sol.t, sol.y.T)
	plt.xlabel('Time (s)')
	plt.show()



dts = [1, 0.5, 0.25, 0.125, 0.0625, 0.0375, \
		0.01875, 0.009285, 0.0046425]

for i in range(len(dts)):
	euler(dts(i))
