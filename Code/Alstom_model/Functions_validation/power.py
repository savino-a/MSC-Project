import matplotlib.pyplot as plt
import numpy as np
import math


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


demand_list = np.arange(-1, 1, 0.001)
v_list = np.arange(0, 220, 1)
cost_list = np.zeros((len(demand_list), len(v_list)))
for i, demand in enumerate(demand_list):
    for j, v in enumerate(v_list):
        cost_list[i, j] = cost(v, demand)
plt.figure(figsize=(10, 6))
plt.imshow(cost_list, extent=(0, 220, -1, 1), aspect="auto", origin="lower")
plt.colorbar(label="Cost")
plt.title("Cost vs Speed and Demand")
plt.xlabel("Speed (km/h)")
plt.ylabel("Demand")
plt.grid()
plt.show()
