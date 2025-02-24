import unittest
import qiskit
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model
from qiskit_ibm_runtime import SamplerV2 as SamplerV22
import numpy as np
from qiskit_ibm_runtime import SamplerV2 as Sampler
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

matplotlib.use("TkAgg")  # Set non-interactive backend
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os


def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


class Qiskit_Problem:
    def __init__(self, N, D, vmax, delta_v=1, alpha=0.5, dist_tolerance=1, p=2):
        self.mip = Model(name="MIP Model")
        self.N = N
        self.delta_v = delta_v
        self.vmax = vmax
        self.D = D
        self.alpha = alpha
        self.dist_tolerance = 1
        self.p = p
        self.tsppb = False
        self._define_variables()
        self._define_cost()
        self._define_constraints()
        self._convert_()
        self._circuit_()
        

    def _define_variables(self):
        self.x, self.y, self.z = {}, {}, {}
        for i in range(self.N):
            self.x[i] = self.mip.binary_var(name=f"x_{i}")
            self.y[i] = self.mip.binary_var(name=f"y_{i}")
            self.z[i] = self.mip.binary_var(name=f"z_{i}")

    def _define_cost(self):
        objective = self.mip.linear_expr()
        for i in range(self.N):
            objective += (self.delta_v**2) * self.x[i] - self.alpha * (
                self.delta_v**2
            ) * self.y[i]
        self.mip.minimize(objective)

    def _define_constraints(self):
        self._distance_constraint_strict(self.N, self.delta_v)
        """self._distance_constraint(self.N, self.delta_v, self.dist_tolerance)"""
        self._net_zero_constraint(self.N)
        self._vmax_constraint(self.N, self.vmax, self.delta_v)
        self._simu_acc_decc_constraint_linear(self.N)
        """self._simu_acc_decc_constraint(self.N)"""

    def _distance_constraint_strict(self, Nc, delta_v):
        distance = self.mip.linear_expr()
        velocity = 0
        for i in range(0, Nc):
            velocity = velocity + delta_v * (self.x[i] - self.y[i])
            distance += velocity
        self.mip.add_constraint(distance == self.D, "Distance_constraint")

    def _distance_constraint(self, Nc, delta_v, tolerance):
        distance = self.mip.linear_expr()
        velocity = 0
        for i in range(0, Nc):
            velocity = velocity + delta_v * (self.x[i] - self.y[i])
            distance += velocity
        self.mip.add_constraint(
            distance <= self.D + tolerance, "Max_Distance_constraint"
        )
        self.mip.add_constraint(
            distance >= self.D - tolerance, "Min_Distance_constraint"
        )

    def _net_zero_constraint(self, Nc):
        self.mip.add_constraint(
            self.mip.sum((self.y[i] - self.x[i]) for i in range(0, Nc)) == 0,
            "Net_Zero_constraint",
        )

    def _vmax_constraint(self, Nc, vmax, delta_v):
        self.mip.add_constraint(
            self.mip.sum((delta_v * (self.x[i]) for i in range(0, Nc))) <= vmax,
            "Maximum_Speed_constraint",
        )

    def _simu_acc_decc_constraint_linear(self, Nc):
        for i in range(0, Nc):
            self.mip.add_constraint(self.z[i] <= self.x[i], f"z_u_d_{i}")
        for i in range(0, Nc):
            self.mip.add_constraint(self.z[i] <= self.y[i], f"z_p_d_{i}")

        for i in range(0, Nc):
            self.mip.add_constraint(
                self.z[i] >= self.x[i] + self.y[i] - 1, f"z_u_p_d_{i}"
            )

        for i in range(0, Nc):
            self.mip.add_constraint(
                self.z[i] == 0,
                "No simultaneous braking or acceleration constraint" + str(i),
            )
        """self.mip.add_constraint(self.mip.sum(self.z[i] for i in range(0, Nc)) == 0 , "No_simultaneous_braking_or_acceleration_constraint")"""

    def _simu_acc_decc_constraint(self, Nc):
        for i in range(0, Nc):
            self.mip.add_constraint(self.x[i] * self.y[i] == 0, "No simu" + str(i))
        """self.mip.add_constraint(self.mip.sum(self.x[i]*self.y[i] for i in range(0, Nc)) == 0 , "No_simultaneous_braking_or_acceleration_constraint")"""

    def _print_model(self):
        print(self.mip.prettyprint())

    def _convert_(self):
        if self.tsppb:
            self.qp_quad = self.mip.to_quadratic_program()
        else:
            self.qp_quad = from_docplex_mp(self.mip)
        conv = QuadraticProgramToQubo()
        self.qubo = conv.convert(self.qp_quad)
        self.qubitOp, self.offset = self.qubo.to_ising()

    def _circuit_(self):
        self.circuit = QAOAAnsatz(cost_operator=self.qubitOp, reps=self.p)
        self.circuit.measure_all()
        self.num_qubits = self.circuit.num_qubits

    def _solve_(self, backend=AerSimulator(), tol=1e-3, method="COBYLA", shots=1000):
        print("Chosen backend is :" + str(backend))
        self._transpile_circuit_(backend)
        self._optimize_circuit_(backend, tol, method, shots)
        self._get_results_(backend, shots)
        self._post_process_()

    def _transpile_circuit_(self, backend):
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        self.trs_circuit = pm.run(self.circuit)

    def _optimize_circuit_(self, backend, tol, meth, shots):
        initial_gamma = np.pi
        initial_beta = np.pi / 2
        init_params = []
        for i in range(self.p):
            init_params.append(initial_gamma)
            init_params.append(initial_beta)

        self.objective_func_vals = []  # Global variable
        with Session(backend=backend) as session:
            # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
            estimator = Estimator(mode=session)
            estimator.options.default_shots = shots
            '''if backend != AerSimulator():
                # Set simple error suppression/mitigation options, this is for when on quantum hardware
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.dynamical_decoupling.sequence_type = "XY4"
                estimator.options.twirling.enable_gates = True
                estimator.options.twirling.num_randomizations = "auto"'''

            result = minimize(
                self._cost_func_estimator_,
                init_params,
                args=(self.trs_circuit, self.qubitOp, estimator),
                method=meth,
                tol=tol,
            )
            self.opt_circuit = self.trs_circuit.assign_parameters(result.x)

    def _cost_func_estimator_(self, params, ansatz, hamiltonian, estimator):
        isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        total_cost = cost + self.offset
        self.objective_func_vals.append(total_cost)
        return total_cost

    def _get_results_(self, backend, shots):
        # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
        """if backend != AerSimulator():
            sampler = Sampler(backend=backend)
            # Set simple error suppression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"
        else:"""
        sampler = SamplerV22(mode=backend)
        sampler.options.default_shots = shots

        pub = (self.opt_circuit,)
        job = sampler.run([pub], shots=int(shots))

        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())
        self.final_distribution_int = {
            key: val / shots for key, val in counts_int.items()
        }
        self.final_distribution_bin = {
            key: val / shots for key, val in counts_bin.items()
        }
        self.sorted_dist_bin = dict(
            sorted(
                self.final_distribution_bin.items(), key=lambda x: x[1], reverse=True
            )
        )

    def _post_process_(self):
        keys = list(self.final_distribution_int.keys())
        values = list(self.final_distribution_int.values())
        most_likely = keys[np.argmax(np.abs(values))]
        most_likely_bitstring = to_bitstring(most_likely, self.num_qubits)
        most_likely_bitstring.reverse()
        self.solution = most_likely_bitstring
        print(self.solution)



qi_pb = Qiskit_Problem(N=3, D=1, vmax=1,p=2)
qi_pb._solve_()
