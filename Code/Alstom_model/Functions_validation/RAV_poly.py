A, B, C, m, v_max, p_max, c_max = 0.632, 40.7, 3900, 320000, 220, 4305220, 320000
import numpy as np


def P(v):
    return A + B * v + C * v**2


a, b = 0, 250
n = 1

# Choose n+1 interpolation points
x_points = np.linspace(a, b, 1000)
y_points_bis = []
for x in x_points:
    y_points_bis.append(P(x))

# Get the polynomial coefficients (highest degree first)
coeffs = np.polyfit(x_points, y_points_bis, deg=n)

# Create polynomial function
Prav = np.poly1d(coeffs)


def Prav_approx(v):
    return Prav(v)


print(Prav_approx(0))
