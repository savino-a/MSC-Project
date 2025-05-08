import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Alstom_Problem:
    def __init__(self, A, B, C, m, v=0, v_max=220, p_max=4305220):
        self.A = A
        self.B = B
        self.C = C
        self.m = m
        self.v = v
        self.v_max = v_max
        self.p_max = p_max
        self.c_max = m * 0.1 * 10
        self.v_list = [v]
        self.demand_list = []
        self.power_usage = 0
        self.distance = 0
        self.distance_list = [0]
        self.power_usage_list = []
        self.acc_list = []

    def traction(self):
        v = self.v
        if 0 <= v <= 50:
            return -354.1 * v + 2.44 * (10**5)
        elif 50 <= v < 60:
            return -881.3 * v + 2.704 * (10**5)
        elif 60 <= v < 220:
            return -0.05265 * (v**3) + 28.78 * (v**2) - 5603 * v + 4.566 * (10**5)
        else:
            return 0

    def braking(self):
        v = self.v
        if 0 <= v <= 20:
            return 9925 * v + 1.243
        if 20 <= v <= 100:
            return 2.039 * (10**-13) * v + 1.985 * (10**5)
        if 100 <= v <= 220:
            return 5.389 * (v**2) - 2583 * v + 4.012 * (10**5)
        else:
            return 0

    def acc(self, demand=1):
        v = self.v
        if demand >= 0:
            return self.acc_pos(v, demand)
        else:
            return self.acc_neg(v, demand)

    def acc_pos(self, v, demand=1):
        F = self.traction() * demand
        F_available = F - self.A * (v**2) - self.B * v - self.C
        acc = F_available / self.m
        return acc * 3.6

    def acc_neg(self, v, demand=1):
        F = self.braking() * demand
        F_available = F - self.A * (v**2) - self.B * v - self.C
        acc = F_available / self.m
        return acc * 3.6

    def speed(self, demand=1):
        acc = self.acc(demand)
        self.acc_list.append(acc)
        self.v += acc
        self.v_list.append(self.v)

    def cost(self, demand=1):
        F = demand
        F_max = 1
        V = self.v
        V_max = self.v_max
        V_safe = min(V, V_max)
        # Ensure V is within a safe range for math.exp
        c = (
            (1 - math.exp(-V_safe / (V_max + 0.001)))
            * math.tanh(100 * F)
            * (1 - math.exp(-abs(F) / (F_max + 0.001)))
        )
        Puti = (c / self.c_max) * self.p_max
        self.power_usage += Puti
        self.power_usage_list.append(Puti)

    def iterate(self, demand):
        vi = self.v
        self.demand_list.append(demand)
        self.speed(demand)
        self.cost(demand)
        self.distance += vi + (self.v - vi) / 2
        self.distance_list.append(self.distance)

    def reset(self):
        self.v = 0
        self.v_list = [0]
        self.demand_list = []
        self.power_usage = 0
        self.distance = 0
        self.distance_list = []
        self.power_usage_list = []
        self.acc_list = []

    def objective(self, demand_list, D, penalty_weight=1000):
        self.reset()
        for demand in demand_list:
            self.iterate(demand)
            if self.distance >= D:
                break
        final_speed_penalty = penalty_weight * abs(self.v)
        return self.power_usage + final_speed_penalty

    def distance_constraint(self, demand_list, D):
        self.reset()
        for demand in demand_list:
            self.iterate(demand)
        return self.distance - D

    def speed_constraint(self, demand_list):
        self.reset()
        for demand in demand_list:
            self.iterate(demand)
            if self.v <= 0:
                return self.v
        return 1  # Ensure positive speed

    def final_speed_constraint(self, demand_list, D, tolerance=0.1):
        self.reset()
        for demand in demand_list:
            self.iterate(demand)
            if self.distance >= D:
                break
        return tolerance - abs(self.v)

    def optimize(self, D, initial_demand_list, bounds, maxiter=10000):
        constraints = [
            {
                "type": "eq",
                "fun": self.distance_constraint,
                "args": (D,),
            },  # Ensure distance is exactly D
            {
                "type": "ineq",
                "fun": self.speed_constraint,
            },  # Ensure positive speed
            {
                "type": "ineq",
                "fun": self.final_speed_constraint,
                "args": (D, 0.1),
            },  # Ensure final speed is close to 0
        ]

        result = minimize(
            self.objective,
            initial_demand_list,
            args=(D,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter},
        )

        return result


# Define the problem
A, B, C, m, v_max, p_max = 0.632, 40.7, 3900, 320000, 220, 4305220
problem = Alstom_Problem(A, B, C, m, v_max=v_max, p_max=p_max)

# Define the distance to travel
D = 100  # Example distance

# Initial guess for the demand list
initial_demand_list = np.ones(200)  # Example initial guess

# Define bounds for the demand list
bounds = [(-1, 1) for _ in range(len(initial_demand_list))]

# Perform the optimization
result = problem.optimize(D, initial_demand_list, bounds)

# Get the optimized demand list
optimized_demand_list = result.x

# Plot the optimal distance and speed profiles
problem.reset()
for demand in optimized_demand_list:
    problem.iterate(demand)
    if problem.distance >= D:
        break

fig, ax1 = plt.subplots()

color = "tab:blue"
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Distance", color=color)
ax1.plot(problem.distance_list, color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = "tab:red"
ax2.set_ylabel("Speed", color=color)  # we already handled the x-label with ax1
ax2.plot(problem.v_list, color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("Optimal Distance and Speed Profiles")
plt.grid(True)
plt.show()

print("Optimized demand list:", optimized_demand_list)
print("Minimum power usage:", result.fun)
