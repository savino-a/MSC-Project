import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D


def cost(V, demand):
    F_max = 1
    V_max = 220
    F = demand
    p_max = 4305220
    c_max = 320000
    V_safe = min(V, V_max)
    # Ensure V is within a safe range for math.exp
    c = (
        (1 - math.exp(-V_safe / (V_max + 0.001)))
        * math.tanh(100 * F)
        * (1 - math.exp(-abs(F) / (F_max + 0.001)))
    )
    Puti = (c / c_max) * p_max
    return Puti


# Define the domain
a_x, b_x = 0, 250
a_y, b_y = -1, 1
n = 1  # degree in x
m = 1  # degree in y

# Create the interpolation points (equally spaced)
x_points = np.linspace(a_x, b_x, 1000)
y_points = np.linspace(a_y, b_y, 1000)
X, Y = np.meshgrid(x_points, y_points, indexing="ij")
Z = np.eye(1000)
for i in range(len(x_points)):
    for j in range(len(y_points)):
        x = x_points[i]
        y = y_points[j]
        Z[i, j] = cost(x, y)


# Build the Vandermonde matrix for 2D
x_flat = X.ravel()
y_flat = Y.ravel()

A = np.vstack([(x_flat**i) * (y_flat**j) for i in range(n + 1) for j in range(m + 1)]).T

b = Z.ravel()
coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

print(f"Coefficients of the polynomial: {coeffs}")


# Define the polynomial function
def P(x, y):
    val = 0
    idx = 0
    for i in range(n + 1):
        for j in range(m + 1):
            val += coeffs[idx] * (x**i) * (y**j)
            idx += 1
    return val


# Evaluation grid
x_eval = x_points
y_eval = y_points
X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing="ij")
Z_eval = P(X_eval, Y_eval)

# Plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(X_eval, Y_eval, Z, cmap="viridis")
ax.set_title("Original function")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(X_eval, Y_eval, Z_eval, cmap="plasma")
ax2.set_title("Polynomial approximation")

plt.show()
