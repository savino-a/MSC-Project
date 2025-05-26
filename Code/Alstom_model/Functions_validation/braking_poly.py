import numpy as np
import matplotlib.pyplot as plt

# Define your function


def f(v):
    if 0 <= v <= 20:
        return 9925 * v + 1.243
    if 20 <= v <= 100:
        return 2.039 * (10**-13) * v + 1.985 * (10**5)
    if 100 <= v <= 220:
        return 5.389 * (v**2) - 2583 * v + 4.012 * (10**5)
    else:
        return 0


# Interval [a, b] and degree n
a, b = 0, 250
n = 0

# Choose n+1 interpolation points
x_points = np.linspace(a, b, 1000)
y_points_bis = []
for x in x_points:
    y_points_bis.append(f(x))

# Get the polynomial coefficients (highest degree first)
coeffs = np.polyfit(x_points, y_points_bis, deg=n)

print(f"Coefficients of the F_braking polynomial: {coeffs}")

# Create polynomial function
p = np.poly1d(coeffs)


def acc_neg(v, demand=1):
    A, B, C, m, v_max, p_max = 0.632, 40.7, 3900, 320000, 220, 4305220
    F = f(v) * demand
    F_available = F + A * (v**2) + B * v + C
    acc = F_available / m
    return acc * 3.6


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
        Z[i, j] = acc_neg(x, y)


# Build the Vandermonde matrix for 2D
x_flat = X.ravel()
y_flat = Y.ravel()

A = np.vstack([(x_flat**i) * (y_flat**j) for i in range(n + 1) for j in range(m + 1)]).T

b = Z.ravel()
coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)


# Define the polynomial function
def Pdecc(x, y):
    val = 0
    idx = 0
    for i in range(n + 1):
        for j in range(m + 1):
            val += coeffs[idx] * (x**i) * (y**j)
            idx += 1
    return val


print(f"Coefficients of the power polynomial: {coeffs}")

# Plotting
if __name__ == "__main__":
    x_plot = x_points
    plt.plot(x_plot, y_points_bis, label="Original function")
    plt.plot(x_plot, p(x_plot), label=f"Polynomial degree {n}")
    plt.legend()
    plt.grid(True)
    plt.show()
