# Representing the acceleration/braking problem

## Objective

The objective of the problem is to minimize the amount of energy used by a train to move from point A to point B. In order to do this we must figure the optimal time when the train should be accelerating, coasting or braking.

## First formulation

In order to do this we will model the train as being in one of three states during the trip.

- State 1: Acceleration where: $\vec{a}=a*\vec{u}$ between t=0 and t=$t_{1}$
- State 2: Coasting where: $\vec{a}=0$ between t=$t_{1}$ and t=$t_{2}$
- State 3: Braking where: $\vec{a}= -a'$ between t=$t_{2}$ and t=T

In this model we have made a few hypothesis:

- There is no friction
- We do not require energy for braking
- There is no regenerative braking
- The efficiency ($\eta$) of the motors is constant

This gives us that:

```math

v(t) = 
\begin{cases} 
at & \text{for } 0 < t < t_1 \\ 
v_{\text{max}} & \text{for } t_1 < t < t_2 \\ 
v_{\text{max}} - (t - t_1) a' & \text{for } t_2 < t < T 
\end{cases}
```
![Plot1](../Illustrations/newplot.png)
From this we can determine that:
$$
E_{total} = E_{State 1} + E_{State 2} + E_{State 3} \\
And \ since\ : \ E_{State 2} = E_{State 3} = 0 \\
\Rightarrow  E_{total} = E_{State 1} \\
And \ : \ E_{State 1}=E(t_{1})-E(t=0) = E(t_{1})
$$
Considering only the cinetic energy, we need to minimize:
$$
E_{total} = \frac{\eta*m*v_{max}^2}{2}
$$
We will not consider the mass of the train in our problem as it is a constant.

## Converting the problem to a QUBO problem

### What are QUBO problems

The QUBO (Quadratic Unconstrained Binary Optimization) framework is a mathematical formulation used in optimization problems, particularly in quantum computing when the objective is to minimize a quadratic function of binary variables.

#### Particularities

1. **Quadratic Objective Function**: The objective function is expressed as a quadratic polynomial of the binary variables. It typically has the form:

```math
   f(x) = \sum_{i} a_i x_i + \sum_{i < j} b_{ij} x_i x_j
```

   Where:

- $a_{i}$ are linear coefficients
- $b_{ij}$ are quadratic coefficients
- $x_i$ are the binary variables.

2. **Unconstrained**: QUBO problems do not have explicit constraints on the variables, although constraints can often be embedded into the objective function.

### Converting our problem to QUBO
