from qiskit_optimization import QuadraticProgram
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer import AerSimulator
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
from docplex.mp.relax_linear import LinearRelaxer
import matplotlib.pyplot as plt
import numpy as np
from docplex.mp.basic import Expr
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.util.status import JobSolveStatus
import cplex.callbacks as cpx_cb
from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin
from docplex.mp.model import Model
from docplex.mp.relax_linear import LinearRelaxer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import IntegerToBinary
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import Sampler
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import (
    CobylaOptimizer,
    CplexOptimizer,
    MinimumEigenOptimizer,
)
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import (
    QuadraticProgramToQubo,
    QuadraticProgramConverter,
)
from qiskit_optimization.converters import InequalityToEquality

opt_model = Model(name="MIP Model")
Nc = 5  # Nc is the number of seconds
Dist = 2  # Distance to travel
tolerance = 2  # Tolerance in distance travelled
delta_v = 1  # Rate of acceleration/deceleration set to 1
vmax = 150  # Max speed of a TGV in France (in m/s)
alpha = 0.05  # Regenerative braking efficiency

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

## Objective is the hamiltonian/energy value we want to minimize
## Energy:
for i in range(0, Nc):
    objective += (delta_v**2) * x[i] - alpha * (delta_v**2) * y[i]
    # objective += (delta_v**2)*x[i]
opt_model.minimize(objective)

## Constraints

# Constraint 1: Simultaneous acc/decc
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

# Constraint 2: Distance constraint
distance = opt_model.linear_expr()
velocity = 0
for i in range(0, Nc):
    velocity = velocity + delta_v * (x[i] - y[i])
    distance += velocity

"""opt_model.add_constraint(distance == Dist, "Distance_constraint")"""

opt_model.add_constraint(distance <= Dist + tolerance, "Max_Distance_constraint")
opt_model.add_constraint(distance >= Dist - tolerance, "Min_Distance_constraint")

# Constraint 3: Net-Zero constraint
opt_model.add_constraint(
    opt_model.sum((y[i] - x[i]) for i in range(0, Nc)) == 0, "Net_Zero_constraint"
)

# Constraint 4: Maximum Speed
opt_model.add_constraint(
    opt_model.sum((delta_v * x[i]) for i in range(0, Nc)) <= vmax,
    "Maximum_Speed_constraint",
)

print(opt_model.prettyprint())

# Conversion to qad_model
qp_quad = from_docplex_mp(opt_model)
print("Number of variables:", len(qp_quad.variables))

ineq2eq = InequalityToEquality()
qp_eq = ineq2eq.convert(qp_quad)


# Conversion to qubo
conv = QuadraticProgramToQubo()
qp = conv.convert(qp_eq)

# Configuring solver
meo = MinimumEigenOptimizer(QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=1000)))

# Solving
result_qaoa = meo.solve(qp)
print(result_qaoa.x)
print(result_qaoa.fval)
