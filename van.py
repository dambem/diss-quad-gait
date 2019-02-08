import time
import numpy as np
import datetime
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


ts = np.linspace(0.0, 10.0, 1000)

osc = odeint(van_der_pol_oscillator_deriv, [1, 1], ts)
plt.figure(figsize=(15,15))
osc = odeint(van_der_pol_oscillator_deriv, [1, 1], ts)
plt.subplot(2, 1, 1)
plt.ylabel("Oscillator Output (x)")
plt.xlabel("Derivative Of Output (x')")
plt.title("Phase Plot of Van Der Pol Oscillator")
plt.plot(osc[:,0], osc[:,1])

plt.subplot(2, 1, 2)
plt.title("Graph Of Oscillator Output (x) as a function of time")
plt.xlabel("Time T (s)")
plt.ylabel("Oscillator Output (x)")
plt.plot(ts, osc[:,1])



plt.savefig('vanderpol'+datetime.datetime.today().strftime('%H-%M-%s-%d-%m-%y'), dpi=250)

plt.show()
