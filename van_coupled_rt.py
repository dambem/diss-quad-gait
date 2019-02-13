import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


matrix = [[0,-0.2,-0.2,-0.2],
          [-0.2, 0, -0.2, -0.2],
          [-0.2, -0.2, 0, -0.2],
          [-0.2, -0.2, -0.2, 0]]
def van_der_pol_oscillator_deriv(x, t):
    x0 = x[1]
    x1 = mu * ((p - (x[0] ** 2.0)) * x0) - x[0]*w
    res = np.array([x0, x1])
    return res


start_time = time.time()
prev_time = time.time()
count = 1

# forcing frequency
k2 = 1
# amplitude
k1 = 1
#  offset parameter
q1 = 1
# feedback
feed = 0
# def van_der_pol_coupled(x, t):
#     right_side = q1 + k1*np.sin(k2*t) + feed

# osc= odeint(van_der_pol_oscillator_deriv, [start_y, start_x], [count-0.09, count])
start_y = [1,1,1,1]
start_x = [0,0,0,0]
while (time.time() - start_time <= 50):
    count+=0.01
    # if (count > (start_count+6)):
    #     start_count = count
    #  Leaky integrator - to fix the issues
    current_time = time.time()
    mu = 1
    p = 2
    w = 20
    x_list  = []
    for i in range(4):
        osc= odeint(van_der_pol_oscillator_deriv, [start_y[i], start_x[i]], [count-0.01, count])
        x = osc[1][1]
        y = osc[1][0]
        x_list.append(x)
        start_y[i] = y
        start_x[i] = x
        plt.subplot(2, 1, 1)
        plt.scatter(y, x)
        plt.subplot(2, 1, 2)
        plt.scatter(current_time, x)
        plt.pause(0.0001)

    #osc_list.append[osc]# for i in range(4):



    print(x_list)
    # osc2 = odeint(osc, [start_y, start_x], [count-0.09, count])

    # # prev_time = time.time()
    # # print(osc)
    # start_y = osc[1][0]
    # start_x = osc[1][1]


plt.show()
