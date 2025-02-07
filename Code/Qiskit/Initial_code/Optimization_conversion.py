# # This is the code for Quantum Optimization
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model
import numpy as np
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as SamplerV2
from qiskit.primitives import Sampler, Estimator
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz
import qiskit_aer as Aer
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.converters import InequalityToEquality
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import matplotlib
import csv

matplotlib.use("TkAgg")  # Set non-interactive backend
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os
import time


# ## We first create the Docplex/Cplex optimization problem


opt_model = Model(name="MIP Model")

Nc = 3  # Nc is the number of seconds

Dist = 1  # Distance to travel

tolerance = 0  # Tolerance in distance travelled

delta_v = 1  # Rate of acceleration/deceleration set to 1

vmax = 1  # Max speed of a TGV in France (in m/s)

alpha = 0.05  # Regenerative braking efficiency

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

#### Print the optimization model info

"""opt_model.print_information()
opt_model.prettyprint()"""


"""
Constraint 2: (Total Distance constraints)
"""

distance = opt_model.linear_expr()
velocity = 0
for i in range(0, Nc):
    velocity = velocity + delta_v * (x[i] - y[i])
    distance += velocity
"""opt_model.add_constraint(distance <= Dist+tolerance, "Max_Distance_constraint")
opt_model.add_constraint(distance >= Dist-tolerance, "Min_Distance_constraint")"""
opt_model.add_constraint(distance == Dist, "Distance_constraint")

#### Print the optimization model info

"""opt_model.print_information()"""


"""
Constraint 3: (Net-Zero contraint)

"""
opt_model.add_constraint(
    opt_model.sum((y[i] - x[i]) for i in range(0, Nc)) == 0, "Net_Zero_constraint"
)

#### Print the optimization model info

"""opt_model.print_information()"""


"""
Constraint 4: (Maximum Speed)

"""
opt_model.add_constraint(
    opt_model.sum((delta_v * x[i]) for i in range(0, Nc)) <= vmax,
    "Maximum_Speed_constraint",
)

#### Print the optimization model info

"""opt_model.print_information()"""


# Constraint 5: (Positive Speed)
for i in range(Nc):
    opt_model.add_constraint(
        opt_model.sum((x[i] - y[i]) for i in range(0, i)) >= 0,
        "Positive_Speed_constraint" + str(i),
    )

    #### Print the optimization model

"""opt_model.print_information()"""

#### Print the optimization model
"""print(opt_model.prettyprint())"""


# ## Problem conversion


# Conversion to qad_model
qp_quad = from_docplex_mp(opt_model)
"""print("Number of variables:", len(qp_quad.variables))"""


ineq2eq = InequalityToEquality()
qp_eq = ineq2eq.convert(qp_quad)


# Conversion to qubo
conv = QuadraticProgramToQubo()
qubo = conv.convert(qp_eq)

"""print(f"QUBO variables: {len(qubo.variables)}")
print(f"QUBO constraints: {len(qubo.linear_constraints)}")"""


qubitOp, offset = qubo.to_ising()
"""print("Offset:", offset)
print("Ising Hamiltonian:")
print(str(qubitOp))"""
num_qubits = qubitOp.num_qubits
"""print(num_qubits)"""


qubo_circuit = QAOAAnsatz(cost_operator=qubitOp, reps=3)
qubo_circuit.measure_all()

"""qubo_circuit.draw("mpl")"""


qubo_circuit.parameters


# ### Optimize circuit


# QiskitRuntimeService.save_account(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>", overwrite=True, set_as_default=True)
"""service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(min_num_qubits=num_qubits+5)"""
backend = AerSimulator()
"""print(backend)"""

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

candidate_circuit = pm.run(qubo_circuit)
"""candidate_circuit.draw()"""


# ### Execute circuit


initial_gamma = np.pi
initial_beta = np.pi / 2
init_params = [
    initial_gamma,
    initial_beta,
    initial_gamma,
    initial_beta,
    initial_gamma,
    initial_beta,
]


def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)

    return cost


objective_func_vals = []  # Global variable
with Session(backend=backend) as session:
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options, this is for when on quantum hardware
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, qubitOp, estimator),
        method="COBYLA",
        tol=1e-3,
    )
    """print(result)"""


"""plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()"""


optimized_circuit = candidate_circuit.assign_parameters(result.x)


from qiskit_ibm_runtime import SamplerV2 as Sampler

# If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
sampler = Sampler(mode=backend)
sampler.options.default_shots = 10000

# Set simple error suppression/mitigation options
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
sampler.options.twirling.enable_gates = True
sampler.options.twirling.num_randomizations = "auto"

pub = (optimized_circuit,)
job = sampler.run([pub], shots=int(1e4))

counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_100_int = {key: val / shots for key, val in counts_int.items()}

final_distribution_int = {key: val / shots for key, val in counts_int.items()}
final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
"""print(final_distribution_int)
print(final_distribution_bin)"""


# Sort both distributions by probability in descending order
sorted_dist_int = dict(
    sorted(final_distribution_int.items(), key=lambda x: x[1], reverse=True)
)
sorted_dist_bin = dict(
    sorted(final_distribution_bin.items(), key=lambda x: x[1], reverse=True)
)

# Print top 10 most likely outcomes
print("\nTop 10 outcomes (binary representation):")
for k, v in list(sorted_dist_bin.items())[:5]:
    """print(f"{k}: {v}")"""


# ### Post-process


# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, num_qubits)
most_likely_bitstring.reverse()

"""print("Result bitstring:", most_likely_bitstring)"""


x_value = most_likely_bitstring[0:Nc]
y_value = most_likely_bitstring[Nc : 2 * Nc]


# ### Visualization


def distance(x, y, Nc):
    velocity = 0
    vel = [0]
    dist = [0]
    dist_tot = 0
    for i in range(0, Nc):
        velocity = velocity + delta_v * (x[i] - y[i])
        vel.append(velocity)
        dist_tot += velocity
        dist.append(dist_tot)
    return dist, vel


time = np.arange(Nc + 1)
distn, velo = distance(x_value, y_value, Nc)
