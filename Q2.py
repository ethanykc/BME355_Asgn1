# BME355 Assignment 1, Question 1

# b) Simulate this oscillator using Euler’s method, with a time step of 0.05s, for 10s. \
# Plot the state vs. time. Write all the code from scratch using Numpy.

import numpy as np
import matplotlib.pyplot as plt


a, b = 1000, 1
alpha = 20000
p0 = 2000
g = 9.807
m = 220

def f(x):
    return np.array([
        alpha * (x[1] + b*(p0-x[0])/(x[0]+a)),
        g-x[0]/m,
        x[1]
    ])

def expl_trape():
    dt = 0.0046425
    time = np.arange (0, 2, dt)
    x= np.array([0,0,0])
    x_trajectory = []
    for t in time:  
        xa = x + dt * f(x)
        xb = xa + dt * f(xa)
        x = x + 0.5 * dt * f(xa +xb)
        x_trajectory.append(x)


    x_trajectory = np.array(x_trajectory)
    plt.subplot(3,1,1)
    plt.plot(time, x_trajectory[:,0] )
    plt.ylabel('Tension (N)')
    plt.subplot(3,1,2)
    plt.plot(time, x_trajectory[:,1])
    plt.ylabel('Velocity (m/s)')
    plt.subplot(3,1,3)
    plt.plot(time,x_trajectory[:,2])
    plt.show()


expl_trape()
