from Derivatives import Derivatives
import numpy as np
import matplotlib.pyplot as plt

Der12 = Derivatives(order = 1, accuracy_order = 2)

x = np.arange(-10,10,0.01)

y = np.sin(x)**2
y_der_exact = 2*np.sin(x)*np.cos(x)

calc_y_der = Der12.calculate_derivative(x,y)
y_der = calc_y_der.derivative
is_, ie_ = calc_y_der.get_extremes()

plt.figure(1)
plt.plot(x, y_der_exact)

plt.plot(x, y_der)

y = np.sin(x)**4
y_der_exact = 4*np.sin(x)**3*np.cos(x)
calc_y_der = Der12.calculate_derivative(x,y)
y_der = calc_y_der.derivative

plt.figure(2)
plt.plot(x, y_der)
plt.plot(x, y_der_exact)

plt.show()
