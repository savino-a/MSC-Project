import unittest
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

matplotlib.use("TkAgg")  # Set non-interactive backend
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.visualization import circuit_drawer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os
