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
import random
from math import sqrt
import time


def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


random.seed(42)


class TSP_DWave_Problem:
    def __init__(self, N, city_coords=None):
        self.bqm = BinaryQuadraticModel("BINARY")
        self.N = N
        self.lagrange_mult = 1000
        if city_coords == None:
            self._build_cities()
        else:
            self.city_coords = city_coords
        self._build_distance_matrix()
        self._define_variables()
        self._define_cost()
        self._define_constraints()

    def _build_cities(self):
        self.city_coords = []
        coords = read_coords("/home/maop7/Github/MSC-Project/Code/TSP/coords.txt")
        print(len(coords))
        for i in range(self.N):
            """x = random.random()
            y = random.random()"""
            [x, y] = coords[i]
            self.city_coords.append((x, y))
        """print(self.city_coords)"""

    def _build_distance_matrix(self):
        self.dist = []
        for i in range(self.N):
            temp = []
            for j in range(self.N):
                if i == j:
                    temp.append(10**6)
                else:
                    (x1, y1) = self.city_coords[i]
                    (x2, y2) = self.city_coords[j]
                    temp.append(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            self.dist.append(temp)

    def _define_variables(self):
        for i in range(self.N):
            for j in range(self.N):
                self.bqm.add_variable(f"x_{i},{j}", 0.0)

    def _define_cost(self):
        # Initialize the objective function
        objective = {}

        # Add linear terms to the objective function
        for i in range(self.N):
            for j in range(self.N):
                wij = self.dist[i][j]
                for p in range(self.N - 1):
                    self.bqm.add_interaction(f"x_{i},{j}", f"x_{j},{j+1}", wij)

    def _define_constraints(self):
        self._single_visit_constraint()
        self._is_visited_constraint()

    def _single_visit_constraint(self):
        for i in range(self.N):
            termss = [(f"x_{i},{j}", 1.0) for j in range(self.N)]
            self.bqm.add_linear_equality_constraint(
                terms=termss,
                constant=-1.0,
                lagrange_multiplier=self.lagrange_mult,
            )

    def _is_visited_constraint(self):
        for j in range(self.N):
            terms = [(f"x_{i},{j}", 1.0) for i in range(self.N)]
            self.bqm.add_linear_equality_constraint(
                terms=terms,
                constant=-1.0,
                lagrange_multiplier=self.lagrange_mult,
            )

    def extract_tour(self):
        """Extract the tour from the solution"""
        tour = [-1] * self.N
        for var, val in self.best_sample.items():
            if val == 1:
                # Parse variable name like "x_i,j"
                i, j = map(int, var.strip("x_").split(","))
                tour[j] = i
        self.tour = tour

    def solve(self, plot=False):

        sampler = SimulatedAnnealingSampler()
        start = time.time()
        sample_set = sampler.sample(self.bqm, num_reads=8192, num_sweeps=8192)
        end = time.time()
        self.time = end - start
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())
        self.extract_tour()
        self.calculate_tour_distance()
        if plot:
            self.plot_sol()
        else:
            print(self.best_sample)
            print(self.tour)

    def solve_with_LEAP(self, plot=False):
        sampler = LeapHybridSampler()
        start = time.time()
        sample_set = sampler.sample(self.bqm)
        end = time.time()
        self.time = end - start
        self.best_sample = sample_set.first.sample
        self.best_energy = sample_set.first.energy
        self.solution_vector = list(self.best_sample.values())
        self.extract_tour()
        self.calculate_tour_distance()
        if plot:
            self.plot_sol()
        else:
            """print(self.best_sample)
            print(self.tour)"""

    def calculate_tour_distance(self):
        """Calculate the total distance of the tour."""
        total_distance = 0
        for i in range(self.N):
            city_i = self.tour[i]
            city_j = self.tour[(i + 1) % self.N]
            total_distance += self.dist[city_i][city_j]
        self.total_distance = total_distance

    def plot_sol(self):
        """Plot the cities and the tour"""
        tour = self.tour

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot cities
        x_coords = [self.city_coords[i][0] for i in range(self.N)]
        y_coords = [self.city_coords[i][1] for i in range(self.N)]
        plt.scatter(x_coords, y_coords, s=100, c="blue", edgecolor="black")

        # Label each city
        for i in range(self.N):
            plt.annotate(f"{i}", (x_coords[i], y_coords[i]), fontsize=12)

        # Plot the tour
        for i in range(self.N):
            city_i = tour[i]
            city_j = tour[(i + 1) % self.N]
            plt.plot(
                [self.city_coords[city_i][0], self.city_coords[city_j][0]],
                [self.city_coords[city_i][1], self.city_coords[city_j][1]],
                "r-",
            )

        plt.title(f"TSP Solution for {self.N} cities")
        plt.grid(True)
        plt.show()


times = []
distances = []
if __name__ == "__main__":
    for i in range(5):
        print("Iteration: ", i)
        a = TSP_DWave_Problem(N=50)
        sol = a.solve_with_LEAP(plot=False)
        times.append(a.time)
        distances.append(a.total_distance)
        print(a.tour)

print("Average time for 100 cities: ", sum(times) / len(times))
print("Average distance for 100 cities: ", min(distances))
