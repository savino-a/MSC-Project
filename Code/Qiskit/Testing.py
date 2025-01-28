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


qubo = QuadraticProgram()
qubo.binary_var("x")
qubo.binary_var("y")
qubo.binary_var("z")
qubo.minimize(
    linear=[1, -2, 3], quadratic={("x", "y"): 1, ("x", "z"): -1, ("y", "z"): 2}
)
qubo.linear_constraint(linear={"x": 1, "y": 2}, sense=">=", rhs=3, name="lin_leq")
print(qubo.prettyprint())

qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qubo)
op, offset = qubo.to_ising()

backend = AerSimulator(method="statevector")
qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
qaoa_result = qaoa.solve(qubo)
print(qaoa_result.prettyprint())
