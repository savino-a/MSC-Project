import numpy as np
import matplotlib.pyplot as plt
import math


def traction(v):
    if 0 <= v <= 50:
        return -354.1 * v + 2.44 * (10**5)
    elif 50 <= v < 60:
        return -881.3 * v + 2.704 * (10**5)
    elif 60 <= v < 220:
        return -0.05265 * (v**3) + 28.78 * (v**2) - 5603 * v + 4.566 * (10**5)
    else:
        return 0


def braking(v):
    if 0 <= v <= 20:
        return 9925 * v + 1.243
    if 20 <= v <= 100:
        return 2.039 * (10**-13) * v + 1.985 * (10**5)
    if 100 <= v <= 220:
        return 5.389 * (v**2) - 2583 * v + 4.012 * (10**5)
    else:
        return 0


speeds = np.arange(0, 250, 0.1)

tractions = [traction(v) for v in speeds]
brakings = [-braking(v) for v in speeds]
plt.plot(speeds, tractions, label="Traction")
plt.plot(speeds, brakings, label="Braking")
plt.title("Traction and Braking Forces vs Speed")
plt.xlabel("Speed (km/h)")
plt.ylabel("Force (N)")
plt.legend()
plt.grid()
plt.show()
