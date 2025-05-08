# Import all


import logging
import numpy as np
from docplex.mp.model_reader import ModelReader
import math
import numpy as np
from docplex.mp.basic import Expr
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.util.status import JobSolveStatus
import cplex.callbacks as cpx_cb
import os
import sys
from docplex.mp.model import Model
from docplex.mp.relax_linear import LinearRelaxer
from docplex.mp.model import Model

import Convoy_example1
from Convoy_example1 import Convoy1

delta_v = Convoy1.delta_vs
speeds = Convoy1.speeds


def find_closest_acceleration(speeds, v):
    # Find the index of the speed closest to v
    closest_index = min(range(len(speeds)), key=lambda i: abs(speeds[i] - v))
    return delta_v[closest_index]


opt_model = Model(name="MIP Model")

Nc = 40
## Nc is the number of seconds I have to go accross track
Dist = 200
## Distance to travel
tol = 10
## Tolerance of Distance


delta_v_acc = 1
delta_v_decc = 2

## Rate of acceleration/deceleration set to 1
# vmax= 89
## Max speed of a TGV in France (in m/s)
vmax = 30
alpha = 0.05
## Regenerative braking efficiency
"""
We define two binary variables for two bits. 
When x=0 and y=0 then constant velocity
When x=1 and y=0 then acceleration
When x=0 and y=1 then breaking

"""
x = {}
for i in range(0, Nc):
    x[i] = opt_model.binary_var(name=f"x_{i}")

y = {}
for i in range(0, Nc):
    y[i] = opt_model.binary_var(name=f"y_{i}")

z = {}
for i in range(0, Nc):
    z[i] = opt_model.binary_var(name=f"z_{i}")


objective = opt_model.linear_expr()
## objective is the hamiltonian/energy value we want to minimize
## Energy:
velocity = 0
for i in range(0, Nc):
    delta_v_acc = find_closest_acceleration(speeds, velocity)
    velocity = velocity + delta_v_acc * x[i] - delta_v_decc * y[i]
    delta_v_acc = find_closest_acceleration(speeds, velocity)
    objective += (delta_v_acc**2) * x[i] - alpha * (delta_v_decc**2) * y[i]
    # objective += (delta_v**2)*x[i]


opt_model.minimize(objective)


## Constraint 1: (simultaneous braking/acceleration)
"""
Constraint 1: (simultaneous braking/acceleration for each time slice)
which means 
for i in range(0, Nc):
    opt_model.add_constraint(x[i] * y[i] == 0 ) 
This will lead to quadratic non-convex constraint. So We linearize it with the above prescription

"""
# opt_model.add_constraint(opt_model.sum(x[i] * y[i] for i in range(1, Nc)) == 0 , "No_simultaneous_braking_or_acceleration_constraint")

for i in range(0, Nc):
    opt_model.add_constraint(z[i] <= x[i], f"z_u_d_{i}")

for i in range(0, Nc):
    opt_model.add_constraint(z[i] <= y[i], f"z_p_d_{i}")

for i in range(0, Nc):
    opt_model.add_constraint(z[i] >= x[i] + y[i] - 1, f"z_u_p_d_{i}")

opt_model.add_constraint(
    opt_model.sum(z[i] for i in range(0, Nc)) == 0,
    "No_simultaneous_braking_or_acceleration_constraint",
)

### This is another way to write the No simultaneous braking or acceleration constraint also
# for i in range(0, Nc):
#     opt_model.add_constraint(z[i] == 0 , f"z_u_p_d_0{i}")

"""
Constraint 2: (Total Distance constraint)
"""

distance = opt_model.linear_expr()
velocity = 0
for i in range(0, Nc):
    delta_v_acc = find_closest_acceleration(speeds, velocity)
    velocity = velocity + delta_v_acc * x[i] - delta_v_decc * y[i]
    distance += velocity
opt_model.add_constraint(distance == Dist, "Distance_constraint")

### old distance constraints for conparison
# opt_model.add_constraint(opt_model.sum(((Nc-i)*delta_v*y[i]-(Nc-i)*delta_v*x[i]) for i in range(0, Nc)) == Dist , "Distance constraint")
# opt_model.add_constraint(opt_model.sum((i*delta_v*y[i]+i*delta_v*x[i]+ ) for i in range(0, Nc)) == Dist , "Distance constraint")

"""
Constraint 3: (Net-Zero contraint)

"""
velocity = opt_model.linear_expr()
for i in range(0, Nc):
    delta_v_acc = find_closest_acceleration(speeds, velocity)
    velocity += delta_v_acc * x[i] - delta_v_decc * y[i]
opt_model.add_constraint(velocity == 0, "Net_Zero_constraint")

"""opt_model.add_constraint(
    opt_model.sum((y[i] * delta_v_decc - delta_v_acc * x[i]) for i in range(0, Nc))
    == 0,
    "Net_Zero_constraint",
)"""

"""
Constraint 4: (Maximum Speed)

"""
max_velocity = opt_model.linear_expr()
for i in range(0, Nc):
    delta_v_acc = find_closest_acceleration(speeds, velocity)
    max_velocity += delta_v_acc * x[i]
opt_model.add_constraint(max_velocity <= Convoy1.calculate_vmax(), "vmax_constraint")

"""opt_model.add_constraint(
    opt_model.sum((delta_v_acc * x[i]) for i in range(0, Nc)) <= vmax,
    "Maximum_Speed_constraint",
)"""


#### Print the optimization model

opt_model.print_information()

print(opt_model.prettyprint())


# %%
result = opt_model.solve(log_output=True)  # (log_output=self.solver_config.cplex_log)
x_value = []
for l in range(0, Nc):
    x_value.append(result.get_value(f"x_{l}"))
    print(f"x_{l} =", result.get_value(f"x_{l}"))

y_value = []
for l in range(0, Nc):
    y_value.append(result.get_value(f"y_{l}"))
    print(f"y_{l} =", result.get_value(f"y_{l}"))


print(
    "Binary Variables X",
    x_value,
    "Binary Variables y",
    y_value,
    "Objective value",
    result.objective_value,
)


# %% [markdown]
# ## Visualisation of results


# %%
def distance(x, y, Nc):
    velocity = 0
    vel = [0]
    dist = [0]
    dist_tot = 0
    for i in range(0, Nc):
        velocity = velocity + delta_v_acc * x[i] - delta_v_decc * y[i]
        vel.append(velocity)
        dist_tot += velocity
        dist.append(dist_tot)
    return dist, vel


# %%
time = np.arange(Nc + 1)
distn, velo = distance(x_value, y_value, Nc)

# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

matplotlib.rcParams.update({"font.size": 15})

# plt.step(time, velo, c='b', marker="o", markersize=1, linestyle='-')#, label='label')
plt.plot(time, velo, c="b", marker="o", markersize=2, linestyle="-")  # , label='label)

plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity vs Time")
plt.grid(axis="x")
plt.grid(axis="y")
plt.legend()
plt.show()

# %% [markdown]
# #### The maximum velocity might seems low above but for 60 units of time total distance covered 700 units is on an average 12 units of velocity

# %%
import matplotlib

plt.figure(figsize=(10, 6))

matplotlib.rcParams.update({"font.size": 15})

plt.plot(
    time, distn, c="b", marker="o", markersize=1, linestyle="-"
)  # , label='label')
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Distance vs Time")
plt.legend()
plt.show()
