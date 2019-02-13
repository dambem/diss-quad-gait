import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu = 1
p = 2
w = 20

def van_der_pol_oscillator_deriv(x, t):
    x0 = x[1]
    x1 = mu * ((p - (x[0] ** 2.0)) * x0) - x[0]*w
    res = np.array([x0, x1])
    return res


start_time = time.time()
prev_time = time.time()
count = 1
start_y = 1
start_x = 1
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


while (time.time() - start_time <= 50):
    count+=0.01
    # if (count > (start_count+6)):
    #     start_count = count
    #  Leaky integrator - to fix the issues
    current_time = time.time()
    osc = odeint(van_der_pol_oscillator_deriv, [start_y, start_x], [count-0.09, count],mxstep=500000)
    plt.subplot(2, 1, 1)
    plt.scatter(osc[1][0], osc[1][1])
    plt.subplot(2, 1, 2)
    plt.scatter(current_time, osc[1][1])
    # prev_time = time.time()
    print(osc)
    start_y = osc[1][0]
    start_x = osc[1][1]
    plt.pause(0.0001)

plt.show()
