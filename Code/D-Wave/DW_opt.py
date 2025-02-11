import dimod
from dimod.binary_quadratic_model import BinaryQuadraticModel
import numpy
import neal
from neal import SimulatedAnnealingSampler
import dwave
from dwave.system.samplers import LeapHybridSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite


class DWave_problem:
    def __init__(self, N, D, vmax, delta_v=1, alpha=0.05):
        self.bqm = BinaryQuadraticModel("BINARY")
        self.N = N
        self.delta_v = delta_v
        self.vmax = vmax
        self.D = D
        self.alpha = alpha
        self._define_variables()
        self._define_constraints()

    def _define_variables(self):
        for i in range(self.N):
            self.bqm.add_variable(f"x_{i:03d}", self.delta_v**2)
            self.bqm.add_variable(f"y_{i:03d}", -self.alpha * (self.delta_v**2))
            self.bqm.add_variable(f"z_{i:03d}", 0)

    def _define_constraints(self):
        """self._simu_acc_decc_constraint()"""
        self._distance_constraint(self.N, self.delta_v)
        self._net_zero_constraint(self.N)
        self._vmax_constraint(self.N, self.vmax)

    def _simu_acc_decc_constraint(self):
        lagrange_multiplier = 100
        for i in range(self.N):
            z_var = f"z_{i:03d}"
            x_var = f"x_{i:03d}"
            y_var = f"y_{i:03d}"
            # \sum_{i,k} a_{i,k} x_{i,k} + constant <= ub
            self.bqm.add_linear_inequality_constraint(
                {z_var: 1, x_var: -1},
                constant=0,
                lagrange_multiplier=lagrange_multiplier,
            )
            self.bqm.add_linear_inequality_constraint(
                {z_var: 1, y_var: -1},
                constant=0,
                lagrange_multiplier=lagrange_multiplier,
            )
            self.bqm.add_linear_inequality_constraint(
                {z_var: -1, x_var: +1, y_var: +1},
                constant=-1,
                lagrange_multiplier=lagrange_multiplier,
            )

        z_vars = {f"z_{i:03d}": 1 for i in range(self.N)}

        # Add constraint that sum of all z variables equals 0
        self.bqm.add_linear_equality_constraint(
            terms=z_vars, constant=0, lagrange_multiplier=lagrange_multiplier
        )

    def _distance_constraint(self, N, delta_v):
        lagrange_multiplier = 100
        terms = {}

        for i in range(self.N):
            terms[f"x_{i:03d}"] = float((self.N - i + 1))
            terms[f"y_{i:03d}"] = float((self.N - i + 1))

        self.bqm.add_linear_equality_constraint(
            terms=terms,
            constant=-self.D / self.delta_v,
            lagrange_multiplier=lagrange_multiplier,
        )
        # Have to finish this constraint

    def _net_zero_constraint(self, N):
        lagrange_multiplier = 100
        terms = {}

        for i in range(self.N):
            terms[f"x_{i:03d}"] = 1
            terms[f"y_{i:03d}"] = 1
        self.bqm.add_linear_equality_constraint(
            terms=terms,
            constant=0,
            lagrange_multiplier=lagrange_multiplier,
        )

    def _vmax_constraint(self, N, vmax):
        lagrange_multiplier = 100
        self.bqm.add_linear_inequality_constraint(
            terms={f"x_{i:03d}": 1 for i in range(N)},
            constant=-vmax,
            lagrange_multiplier=lagrange_multiplier,
        )

    def solve(self, plot=False):
        sampler = SimulatedAnnealingSampler()
        sampler_set = sampler.sample(self.bqm, num_reads=8192, num_sweeps=8192)
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())

        print(" Best Sample ( Solution ) : ", self.best_sample)
        print(" Energy of the best sample : ", self.best_energy)
        print(" = " * 50)

        if plot:
            self._plot_velocity_changes()

    def solve_with_qpu(self, plot=False):
        sampler = LeapHybridSampler()
        sampler_set = sampler.sample(self.bqm)
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())

        print(" Best Sample ( Solution ) : ", self.best_sample)
        print(" Energy of the best sample : ", self.best_energy)
        print(" = " * 50)

        if plot:
            self._plot_velocity_changes()

    def _plot_velocity_changes(self) -> None:

        plt.figure(figsize=(10, 6))

        # Time steps are defined by the length of the solution vector divided by 2 ( since 2 bits per time step )
        time_steps = len(self.solution_vector) // 2
        velocity = [0] * time_steps
        for i in range(time_steps):
            # Extract acceleration ( a ) and braking ( b ) states for the current time step
            a = self.solution_vector[2 * i]
            b = self.solution_vector[2 * i + 1]
            # Update the velocity : increase for acceleration (10) , decrease for braking (01)
            velocity_change = (a - b) * self.delta_v
            if i > 0:
                velocity[i] = velocity[i - 1] + velocity_change
            else:
                velocity[i] = velocity_change

        plt.plot(
            range(time_steps),
            velocity,
            label=f" Velocity ",
            marker=" o ",
            color=" #666666 ",
            markersize=4,
        )
        plt.title(" Velocity vs Time ")
        plt.xlabel(" Time Step ")
        plt.ylabel(" Velocity ( m / s ) ")
        plt.legend()
        plt.grid(True)
        plt.show()


a = DWave_problem(N=3, D=1, vmax=1)
