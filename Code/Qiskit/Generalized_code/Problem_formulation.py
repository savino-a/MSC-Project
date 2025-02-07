from setup import *
from Optimization import MI_optimization_problem

Nc = 3
Dist = 1
tolerance = 0
delta_v = 1
vmax = 1
alpha = 0.05


opt_model = Model(name="MIP Model")

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
for i in range(0, Nc):
    objective += (delta_v**2) * x[i] - alpha * (delta_v**2) * y[i]
    # objective += (delta_v**2)*x[i]


opt_model.minimize(objective)


## Constraint 1: (simultaneous braking/acceleration)

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


## Constraint 2: (Total Distance constraints)

distance = opt_model.linear_expr()
velocity = 0
for i in range(0, Nc):
    velocity = velocity + delta_v * (x[i] - y[i])
    distance += velocity
"""opt_model.add_constraint(distance <= Dist+tolerance, "Max_Distance_constraint")
opt_model.add_constraint(distance >= Dist-tolerance, "Min_Distance_constraint")"""
opt_model.add_constraint(distance == Dist, "Distance_constraint")


## Constraint 3: (Net-Zero contraint)
opt_model.add_constraint(
    opt_model.sum((y[i] - x[i]) for i in range(0, Nc)) == 0, "Net_Zero_constraint"
)


## Constraint 4: (Maximum Speed)
opt_model.add_constraint(
    opt_model.sum((delta_v * x[i]) for i in range(0, Nc)) <= vmax,
    "Maximum_Speed_constraint",
)

## Constraint 5: (Positive Speed)
for i in range(Nc):
    opt_model.add_constraint(
        opt_model.sum((x[i] - y[i]) for i in range(0, i)) >= 0,
        "Positive_Speed_constraint" + str(i),
    )


problem = MI_optimization_problem(name="Trajectory_Optimization", model=opt_model)

problem.ising_conversion()
