import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


lamb = [  [0,-0.2,-0.2,-0.2],
          [-0.2, 0, -0.2, -0.2],
          [-0.2, -0.2, 0, -0.2],
          [-0.2, -0.2, -0.2, 0]]
lamb33 = [  [0,-0.2,-0.2,-0.2],
          [-0.2, 0, -0.2, -0.2],
          [-0.2, -0.2, 0, -0.2],
          [-0.2, -0.2, -0.2, 0]]
lamb2 = [[0, -0.2, 0.2, -0.2],
         [-0.2, 0, -0.2, 0.2],
         [0.2, -0.2, 0, -0.2],
         [-0.2, 0.2, -0.2, 0]]

lamb_control = [[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]]
current_i = 0
def van_der_pol_oscillator_deriv(x, t):
    x0 = x[1]
    x1 = mu * ((p - (x[0] ** 2.0)) * x0) - x[0]*w
    res = np.array([x0, x1])
    return res

def van_der_pol_coupled(x, t):
    x0 = x[1]
    x_ai =x[0]
    for j in range(4):
        x_ai += x[0]-(lamb[current_i][j]*start_x[j])
    # x_ai *= time_step
    # osc = odeint(van_der_pol_oscillator_deriv, [x[0], x[1]], [t-time_step, t])
    x1 =  mu * ((p - (x_ai** 2.0))* x0) - x_ai*w
    res = np.array([x0, x1])
    return res

start_time = time.time()
prev_time = time.time()
time_step = 0.01

count = 0

start_y = [np.pi, np.pi, np.pi, np.pi]
start_x = [0,0,0,0]
new_y = [np.pi, np.pi, np.pi, np.pi]
new_x = [0,0,0,0]
oscillator_values = []
while (count <= 10):
    count+= time_step
    current_time = time.time()
    mu = 1
    p = 2
    w = 20
    x_list  = []
    for i in range(4):
        current_i = i
        osc= odeint(van_der_pol_coupled, [start_y[i], start_x[i]], [count-time_step, count])
        x = osc[1][1]
        y = osc[1][0]
        x_list.append(x)
        new_y[i] = y
        new_x[i] = x
        if (i == 0):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='red')
            plt.subplot(2, 1, 2)
            plt.scatter(count, x, c='red')
        if (i == 1):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='green')
            plt.subplot(2, 1, 2)
            plt.scatter(count, x, c='green')
        if (i == 2):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='blue')
            plt.subplot(2, 1, 2)
            plt.scatter(count, x, c='blue')
        if (i == 3):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='yellow')
            plt.subplot(2, 1, 2)
            plt.scatter(count, x, c='yellow')
    start_y = new_y
    start_x = new_x

    plt.pause(0.0001)



plt.show()
