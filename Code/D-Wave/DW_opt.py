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
import seaborn as sns


class DWave_Problem:
    def __init__(self, N, D, vmax, delta_v=1, alpha=0.05,eff=False,trap=False):
        self.bqm = BinaryQuadraticModel("BINARY")
        self.N = N
        self.delta_v = delta_v
        self.vmax = vmax
        self.trap=trap
        self.eff=eff
        self.D = D
        self.alpha = alpha
        if self.eff:
            self._define_cost_efficiency()
        else:
            self._define_variables()
        self._define_constraints()

    def _define_variables(self):
        for i in range(self.N):
            self.bqm.add_variable(f"x_{i:03d}", self.delta_v**2)
            self.bqm.add_variable(f"y_{i:03d}", -self.alpha * (self.delta_v**2))
            self.bqm.add_variable(f"z_{i:03d}", 0)
    
    def _define_cost_efficiency(self):
        # Initialize the objective function
        objective = {}

        # Add linear terms to the objective function
        for i in range(self.N):
            x_var = f"x_{i:03d}"
            y_var = f"y_{i:03d}"
            objective[(x_var, x_var)] = self.delta_v**2 * (self.N - i)
            objective[(y_var, y_var)] = -self.alpha * (self.delta_v**2) * (self.N - i)

        # Add quadratic terms to the objective function
        for i in range(self.N):
            for j in range(i):
                x_var_i = f"x_{i:03d}"
                y_var_i = f"y_{i:03d}"
                x_var_j = f"x_{j:03d}"
                y_var_j = f"y_{j:03d}"
                objective[(x_var_i, x_var_j)] = self.delta_v**2 * -0.001
                objective[(y_var_i, y_var_j)] = self.delta_v**2 * -0.001

        # Add the objective function to the BQM
        for (var1, var2), bias in objective.items():
            if var1 == var2:
                self.bqm.add_variable(var1, bias)
            else:
                self.bqm.add_interaction(var1, var2, bias)

    def _define_constraints(self):
        if self.trap:
            self._distance_trapeze_constraint(self.N, self.delta_v)
        else:
            self._distance_constraint(self.N,self.delta_v)
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
                terms=[((z_var, 1)), ((x_var), -1)],
                constant=0,
                lagrange_multiplier=lagrange_multiplier,
                label="z_1" + str(i),
            )
            self.bqm.add_linear_inequality_constraint(
                terms=[(z_var, 1), (y_var, -1)],
                constant=0,
                lagrange_multiplier=lagrange_multiplier,
                label="z_2" + str(i),
            )
            self.bqm.add_linear_inequality_constraint(
                terms=[(z_var, -1), (x_var, +1), (y_var, +1)],
                constant=-1,
                lagrange_multiplier=lagrange_multiplier,
                label="z_3" + str(i),
            )

        z_vars = [(f"z_{i:03d}", 1) for i in range(self.N)]

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

    def _distance_trapeze_constraint(self, N, delta_v):
        lagrange_multiplier = 100

        termss = [(f"x_{i:03d}", (N - i + 0.5) * delta_v) for i in range(self.N)] + [
            (f"y_{i:03d}", -(N - i + 0.5) * delta_v) for i in range(self.N)
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
            self._plot_results_()
        return self.best_sample

    def solve_with_LEAP(self, plot=False):
        sampler = LeapHybridSampler()
        sampler_set = sampler.sample(self.bqm)
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())
        if plot:
            self._plot_results_()

    def _plot_results_(self) -> None:
        sns.set(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Time steps are defined by the length of the solution vector divided by 2 ( since 2 bits per time step )
        x = []
        y = []
        velocity = [0]
        d=0
        self.distance=[0]

        for i in range(0, self.N):
            x.append(self.best_sample["x_" + ((str(i))).zfill(3)])
            y.append(self.best_sample["y_" + ((str(i))).zfill(3)])
            velocity.append(velocity[i] + ((x[i] - y[i]) * self.delta_v))
        print(x, y)
        self.velocity=velocity
        for i in range(0, self.N):
            d += velocity[i]
            self.distance.append(d)
        
        # Plot velocity changes
        sns.lineplot(
            x=range(self.N + 1),
            y=velocity,
            marker="o",  # Change marker style
            color="b",  # Change color to blue
            markersize=8,  # Increase marker size
            label="Velocity",
            ax=ax1
        )
        ax1.set_xlabel("Step", fontsize=14)  # Increase font size for labels
        ax1.set_ylabel("Velocity", fontsize=14, color="b")
        ax1.tick_params(axis='y', labelcolor="b")
        ax1.legend(loc="upper left", fontsize=12)  # Increase font size for legend
        ax1.grid(True, linestyle="--", alpha=0.7)  # Add grid lines with custom style

        # Create a second y-axis for distance
        ax2 = ax1.twinx()
        sns.lineplot(
            x=range(self.N + 1),
            y=self.distance,
            marker="o",  # Change marker style
            color="r",  # Change color to red
            markersize=8,  # Increase marker size
            label="Distance",
            ax=ax2
        )
        ax2.set_ylabel("Distance", fontsize=14, color="r")
        ax2.tick_params(axis='y', labelcolor="r")
        ax2.legend(loc="upper right", fontsize=12)  # Increase font size for legend

        plt.title(
            "Velocity and Distance Changes Over Time", fontsize=16
        )  # Add a more descriptive title
        plt.show()


a = DWave_Problem(N=5, D=2, vmax=3)
sol = a.solve(plot=True)
