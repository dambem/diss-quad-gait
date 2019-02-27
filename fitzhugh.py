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

x0 = 2
y0 = 2
k2 = 0
k1 = 0
fa = 0
fb = 1
a = 0.1
b = 0.5
c = 1.5
count = 0
time_step = 0.1
while (time.time() - start_time <= 50):
    count+= time_step
    fci = fa + fb * (k1*np.sin(k2*count))
    x1 = (c * (y0 + x0 + (x0**3)/3 + fci))
    # print(x1)
    y1 = -((x0 - a + b*y0)/c)
    # print(y0)
    x0 = x0+(x1/time_step)
    print(x0)
    y0 = y0+(y1/time_step)
    print(y0)
    #  Leaky integrator - to fix the issues
    # plt.subplot(2, 1, 1)
    plt.scatter(y0, x0)
    # plt.subplot(2, 1, 2)
    # plt.scatter(current_time, osc[1][1])
    plt.pause(0.0001)
plt.show()
