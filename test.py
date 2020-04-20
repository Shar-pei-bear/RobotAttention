import numpy as np
import sdeint
import matplotlib.pyplot as plt

A = np.eye(4, k =2)
print(A)

B = np.diag([1, 1, 1, 1]) # diagonal, so independent driving Wiener processes

tspan = np.linspace(0.0, 10.0, 10001)
x0 = np.array([0, 0, 1, 1])


def f(x, t):
    print(A.dot(x))
    return A.dot(x)


def G(x, t):
    return B


result = sdeint.itoint(f, G, x0, tspan)

plt.plot(tspan, result[:, 0], 'b', label='x(t)')
plt.plot(tspan, result[:, 1], 'g', label='y(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()