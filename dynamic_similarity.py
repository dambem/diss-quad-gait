import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.integrate import odeint
a = 1
u = 2
g = 9.8
h = 2
b = 0.3
# x =


def dynamic_similarity(a, u, g, h, b):
    return (a*(u**2/g*h)**b)
print (dynamic_similarity(a, u, g, h, b))

h = np.linspace(0.0, 10.0, 1000)
for n in h:
    plt.scatter(n, dynamic_similarity(a,u,g,n,b))
    print (n)
plt.show()
