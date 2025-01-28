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


Nc = 5  # Nc is the number of seconds
Ny = 2 * Nc
Dist = 1  # Distance to travel
tolerance = 2  # Tolerance in distance travelled
delta_v = 1  # Rate of acceleration/deceleration set to 1
vmax = 150  # Max speed of a TGV in France (in m/s)
alpha = 0.05  # Regenerative braking efficiency

opt_model = Model(name="MIP Model")

y = {}
for i in range(0, Ny):
    y[i] = opt_model.binary_var(name=f"y_{i}")

objective = opt_model.linear_expr()
## objective is the hamiltonian/energy value we want to minimize

lambda1 = 10000
lambda2 = 100
lambda3 = 1000
lambda4 = 100

## Energy:
for i in range(Nc):
    # objective += (delta_v**2) * y[i*2+1] - alpha*(delta_v**2)*y[i*2]
    objective += (delta_v**2) * y[i * 2 + 1]

## Constraint 1: (simultaneous braking/acceleration)
for i in range(Nc):
    objective += lambda1 * y[i * 2] * y[i * 2 + 1]

## Constraint 2: (Distance)
temp = 0
for i in range(Nc):
    # speedy=0
    # for j in range(i+1):
    #    speedy += (y[i*2+1]-y[i*2])*delta_v
    temp += ((Nc - i) * delta_v * y[i * 2 + 1] - (Nc - i) * delta_v * y[i * 2]) - Dist
    # temp += speedy
objective += lambda2 * ((temp) ** 2)

## Constraint 3: (Net-Zero contraint)
temp = 0
for i in range(Nc):
    temp += y[i * 2] - y[i * 2 + 1]
objective += lambda3 * ((temp) ** 2)

## Constraint 4: (Maximum Speed)
temp = 0
for i in range(Nc):
    temp += delta_v * y[i * 2 + 1]
objective += lambda4 * ((temp - vmax) ** 2)

opt_model.minimize(objective)
print(opt_model.prettyprint())

# Conversion to qad_model
qp_quad = from_docplex_mp(opt_model)
print("Number of variables:", len(qp_quad.variables))


qp_eq = qp_quad


# Conversion to qubo
conv = QuadraticProgramToQubo()
qp = conv.convert(qp_eq)

# Configuring solver
meo = MinimumEigenOptimizer(QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=1000)))

# Solving
result_qaoa = meo.solve(qp)
print(result_qaoa.x)
print(result_qaoa.fval)
