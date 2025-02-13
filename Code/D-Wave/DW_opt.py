import dimod
from dimod.binary_quadratic_model import BinaryQuadraticModel
import numpy
import neal
from neal import SimulatedAnnealingSampler
import dwave
from dwave.system.samplers import LeapHybridSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt


class DWave_Problem:
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
        self._distance_constraint(self.N, self.delta_v)
        self._net_zero_constraint(self.N)
        self._vmax_constraint(self.N, self.vmax)
        self._simu_acc_decc_constraint_linear()

    def _simu_acc_decc_constraint_linear(self):
        lagrange_multiplier = 100
        for i in range(self.N):
            z_var = f"z_{i:03d}"
            x_var = f"x_{i:03d}"
            y_var = f"y_{i:03d}"

            self.bqm.add_linear_inequality_constraint(
                {z_var: 1, x_var: -1},
                constant=0,
                lagrange_multiplier=lagrange_multiplier,
                label="z_1",
            )
            self.bqm.add_linear_inequality_constraint(
                {z_var: 1, y_var: -1},
                constant=0,
                lagrange_multiplier=lagrange_multiplier,
                label="z_2",
            )
            self.bqm.add_linear_inequality_constraint(
                {z_var: -1, x_var: +1, y_var: +1},
                constant=-1,
                lagrange_multiplier=lagrange_multiplier,
                label="z_3",
            )

        z_vars = {f"z_{i:03d}": 1 for i in range(self.N)}

        # Add constraint that sum of all z variables equals 0
        self.bqm.add_linear_equality_constraint(
            terms=z_vars, constant=0, lagrange_multiplier=lagrange_multiplier
        )

    def _simu_acc_decc_constraint_quad(self):
        print("")

    def _distance_constraint(self, N, delta_v):
        lagrange_multiplier = 100

        termss = [(f"x_{i:03d}", (N - i) * delta_v) for i in range(self.N)] + [
            (f"y_{i:03d}", -(N - i) * delta_v) for i in range(self.N)
        ]

        self.bqm.add_linear_equality_constraint(
            terms=termss,
            constant=-self.D,
            lagrange_multiplier=lagrange_multiplier,
        )

    def _net_zero_constraint(self, N):
        lagrange_multiplier = 100

        termss = [(f"x_{i:03d}", 1) for i in range(self.N)] + [
            (f"y_{i:03d}", -1) for i in range(self.N)
        ]

        self.bqm.add_linear_equality_constraint(
            terms=termss,
            constant=0,
            lagrange_multiplier=lagrange_multiplier,
        )

    def _vmax_constraint(self, N, vmax):
        lagrange_multiplier = 100

        termss = [(f"x_{i:03d}", self.delta_v) for i in range(N)]
        self.bqm.add_linear_inequality_constraint(
            terms=termss,
            constant=-vmax,
            lagrange_multiplier=lagrange_multiplier,
            label="Vmax_constraint",
        )

    def solve(self, plot=False):

        sampler = SimulatedAnnealingSampler()
        sample_set = sampler.sample(self.bqm, num_reads=8192, num_sweeps=8192)
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())

        if plot:
            self._plot_velocity_changes()
        return self.best_sample

    def solve_with_LEAP(self, plot=False):
        sampler = LeapHybridSampler()
        sampler_set = sampler.sample(self.bqm)
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())
        if plot:
            self._plot_velocity_changes()

    def _plot_velocity_changes(self) -> None:

        plt.figure(figsize=(10, 6))

        # Time steps are defined by the length of the solution vector divided by 2 ( since 2 bits per time step )
        x = []
        y = []
        velocity = [0]

        for i in range(0, self.N):
            x.append(self.best_sample["x_" + ((str(i))).zfill(3)])
            y.append(self.best_sample["y_" + ((str(i))).zfill(3)])
            velocity.append(velocity[i] + ((x[i] - y[i]) * self.delta_v))

        plt.plot(
            range(self.N + 1),
            velocity,
            label=f" Velocity ",
            marker="o",
            color="r",
            markersize=4,
        )
        plt.title(" Velocity vs Time ")
        plt.xlabel(" Time Step ")
        plt.ylabel(" Velocity ( m / s ) ")
        plt.legend()
        plt.grid(True)
        plt.show()


a = DWave_Problem(N=5, D=2, vmax=3)
sol = a.solve(plot=True)
