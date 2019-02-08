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


# osc = odeint(van_der_pol_oscillator_deriv, [1, 1], ts)

start_time = time.time()
prev_time = time.time()
count = 1
start_count = 1
while (time.time() - start_time <= 50):
    count+=0.09
    # if (count > (start_count+6)):
    #     start_count = count
    #  Leaky integrator - to fix the issues
    current_time = time.time()
    osc = odeint(van_der_pol_oscillator_deriv, [1, 1], [start_time, current_time],mxstep=500000)
    plt.subplot(2, 1, 1)
    plt.scatter(osc[1][0], osc[1][1])
    plt.subplot(2, 1, 2)
    plt.scatter(current_time, osc[1][1])
    # prev_time = time.time()
    plt.pause(0.0001)

plt.show()
