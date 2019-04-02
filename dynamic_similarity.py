import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.integrate import odeint
a = 2.4
u = 2
g = 9.8
h = 2
b = 0.34
# x =

def froude_number(u, g, h):
    return (u**2)/g*h
def dynamic_similarity(a, u, g, h, b, froude):
    # froude = froude_number(u, g, h)
    return (a*(froude)**b)


h = np.linspace(0, 0.3, 50)
h2 = np.linspace(0.3, 30, 50)
plt.ylabel("Relative Stride Length")
plt.xlabel("Froude Number")
for n in h:
    plt.scatter(n, dynamic_similarity(a,u,g,h,b, n), marker='s', c='red')
a= 1.9
b = 0.4
for n in h2:
    plt.scatter(n, dynamic_similarity(a,u,g,h,b, n), c='blue')
plt.show()
