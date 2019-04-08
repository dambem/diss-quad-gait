import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
mu = 1
p = 2
w = 20
pe = 0
integral = 0
start_time = time.time()
startTime = 0
output = 0

def van_der_pol_oscillator_deriv(x, t):
    x0 = x[1]
    x1 = mu * ((p - (x[0] ** 2.0)) * x0) - x[0]*w
    res = np.array([x0, x1])
    return res
def van_der_pol_oscillator_deriv_pure(x, t):
    x0 = x[1]
    x1 = 1 * ((1 - (x[0] ** 2.0)) * x0) - x[0]*1
    res = np.array([x0, x1])
    return res
#
# def van_der_pol(x, y, u, t, dt):
#     x  =
count = 0.09
start_y = 2
start_x = 0

# forcing frequency
k2 = 1
# amplitude
k1 = 1
#  offset parameter
q1 = 1
# feedback
feed = 0

values = []
period = []
time_array = []
counter = 0
while (len(period) < 3):
    osc = odeint(van_der_pol_oscillator_deriv_pure, [start_y, start_x], [count-0.005, count],mxstep=500000)
    integral = 0
    plt.subplot(1, 1, 1)
    values.append(osc[1][1])
    if (len(values) >= 4):
        current = values[counter] - values[counter-1]
        previous = values[counter-1] - values[counter-2]
        if(current >= 0 and previous <= 0):
            period.append([count, osc[1][1]])
    prev_time = time.time()
    time_array.append(count)
    start_y = osc[1][0]
    start_x = osc[1][1]
    count += 0.005
    counter += 1
plt.ylabel("Oscillator x output")
plt.xlabel("Time t")
plt.title("Van Der Pol Oscillator Output Over Time")
plt.plot(time_array,values)
    # plt.pause(0.0001)

plt.scatter(period[0][0], period[0][1], s=100, c='r', marker="x")
plt.scatter(period[1][0], period[1][1], s=100, c='r', marker="x")
period2 = round(period[1][0]-period[0][0], 2)
plt.annotate ('', (period[0][0], period[0][1]), (period[1][0], period[1][1]), arrowprops={'arrowstyle':'<->'})
plt.annotate('T = '+str(period2), xy=(5,-2.5), xycoords='data', xytext=(0, 0), textcoords='offset points')


plt.show()
