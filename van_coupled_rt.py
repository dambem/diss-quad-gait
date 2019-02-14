import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


lamb = [  [0,-0.2,-0.2,-0.2],
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
    x_ai = x[0]
    for j in range(4):
        x_ai += (lamb_control[j][current_i]*start_x[j])
    x1 = mu * ((p - (x_ai** 2.0))* x0) - x_ai*w
    res = np.array([x0, x1])
    return res

start_time = time.time()
prev_time = time.time()
time_step = 0.05

count = 0

start_y = [1,1,1,1]
start_x = [0,0,0,0]
new_y = [1,1,1,1]
new_x = [0,0,0,0]
while (count <= 10):
    w = 20
    count+= time_step
    current_time = time.time()
    mu = 1
    p = 5
    w = 30
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
            plt.scatter(current_time, x, c='red')
        if (i == 1):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='green')
            plt.subplot(2, 1, 2)
            plt.scatter(current_time, x, c='green')
        if (i == 2):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='blue')
            plt.subplot(2, 1, 2)
            plt.scatter(current_time, x, c='blue')
        if (i == 3):
            plt.subplot(2, 1, 1)
            plt.scatter(y, x, c='yellow')
            plt.subplot(2, 1, 2)
            plt.scatter(current_time, x, c='yellow')
    start_y = new_y
    start_x = new_x

    # for i in range(4):
    #     x_ai = x_list[i]
    #     for j in range(4):
    #         x_ai += lamb[i][j]*x_list[j]
    #     osc = odeint(van_der_pol_oscillator_deriv, [start_y[i], x_ai], [count-0.01, count])
        # x = osc[1][1]
        # y = osc[1][0]
        # start_y[i] = y
        # start_x[i] = x
    #osc_list.append[osc]# for i in range(4):

        # plt.pause(0.0001)

    # print(x_list)
    # osc2 = odeint(osc, [start_y, start_x], [count-0.09, count])

    # # prev_time = time.time()
    # # print(osc)
    # start_y = osc[1][0]
    # start_x = osc[1][1]


plt.show()
