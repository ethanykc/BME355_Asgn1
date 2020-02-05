# BME355 Assignment 1, Question 1

# b) Simulate this oscillator using Euler’s method, with a time step of 0.05s, for 10s. \
# Plot the state vs. time. Write all the code from scratch using Numpy.

import numpy as np
import matplotlib.pyplot as plt

# function for the derivative of state vector
def f(x):
	return np.matmul([[0, -1], [1, 0]], x)


# Implementation of Euler's method for given system
def euler():
	dt = 0.05
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

euler()