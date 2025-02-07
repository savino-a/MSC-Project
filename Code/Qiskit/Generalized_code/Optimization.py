from setup import *


class MI_optimization_problem:
    def __init__(
        self,
        model=Model(name="MIP_Model"),
        backend=AerSimulator(),
        reps=2,
        optimization_level=3,
        name="MIP_Model",
    ):
        self.model = model
        self.objective = 0
        self.reps = reps
        self.name = name
        self.backend = backend
        self.optimization_level = optimization_level
        self.init_params = []
        self.objective_func_vals = []

    def print_stats(self):
        self.model.prettyprint()

    def qp_conversion(self):
        self.qp_quad = from_docplex_mp(self.model)

    def ineq_to_eq(self):
        ineq2eq = InequalityToEquality()
        self.qp_eq = ineq2eq.convert(self.qp_conversion())

    def qubo_conversion(self):
        self.qubo = QuadraticProgramToQubo().convert(self.ineq_to_eq())

    def ising_conversion(self):
        self.ineq_to_eq()
        self.qubit_op, self.offset = self.qubo.to_ising()

    def quantum_circuit(self):
        self.ising_conversion(self.qubit_op)
        self.qubo_circuit = QAOAAnsatz(cost_operator=self.qubit_op, reps=self.reps)
        print(self.qubo_circuit.parameters)

    def draw_circuit(self):
        self.qubo_circuit.draw("mpl")

    def initialize_params(self):
        initial_gamma = np.pi
        initial_beta = np.pi / 2
        if self.init_params != []:
            print("Parameters already initialized")
        else:
            for i in range(self.reps):
                self.init_params.append(initial_gamma)
                self.init_params.append(initial_beta)

    def initial_circuit_optimizer(self):
        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=backend
        )
        self.qubo_optimized = pm.run(self.qubo_circuit)

    def cost_func_estimator(self, params, ansatz, hamiltonian, estimator):
        isa_hamiltonian = self.qubit_op.apply_layout(ansatz.layout)
        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        results = job.result()[0]
        cost = results.data.evs
        self.objective_func_vals.append(cost)
        return cost

    def optimize_circuit(self, meth="COBYLA", tole=1e-3, num_shots=1000):
        self.initialize_params()
        print(self.backend)
        with Session(backend=backend) as session:
            # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
            estimator = Estimator(mode=session)
            estimator.options.default_shots = num_shots

            # Set simple error suppression/mitigation options, this is for when on quantum hardware
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"

            result = minimize(
                self.cost_func_estimator,
                self.init_params,
                args=(self.qubo_optimized, self.qubitOp, estimator),
                method=meth,
                tol=tole,
            )
            self.optimized_circuit = self.qubo_optimized.assign_parameters(result.x)
            print(result)

        def extract_results(self, num_shots=1000):
            sampler = Sampler(mode=backend)
            sampler.options.default_shots = num_shots

            # Set simple error suppression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"
            pub = (self.optimized_circuit,)
            job = sampler.run([pub], shots=int(num_shots))

            counts_int = job.result()[0].data.meas.get_int_counts()
            counts_bin = job.result()[0].data.meas.get_counts()
            shots = sum(counts_int.values())
            final_distribution_100_int = {
                key: val / shots for key, val in counts_int.items()
            }

            final_distribution_int = {
                key: val / shots for key, val in counts_int.items()
            }
            final_distribution_bin = {
                key: val / shots for key, val in counts_bin.items()
            }
            print(final_distribution_int)
            print(final_distribution_bin)
            return final_distribution_bin
