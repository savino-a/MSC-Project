import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Alstom_Problem:
    def __init__(self, A, B, C, m, v=0, v_max=220, p_max=4305220):
        self.A = A
        self.B = B
        self.C = C
        self.m = m
        self.v = v
        self.v_max = v_max
        self.p_max = p_max
        self.c_max = m * 0.1 * 10
        self.v_list = [v]
        self.demand_list = []
        self.power_usage = 0
        self.distance = 0
        self.distance_list = [0]
        self.power_usage_list = []
        self.acc_list = []

    def traction(self):
        v = self.v
        if 0 <= v <= 50:
            return -354.1 * v + 2.44 * (10**5)
        elif 50 <= v < 60:
            return -881.3 * v + 2.704 * (10**5)
        elif 60 <= v < 220:
            return -0.05265 * (v**3) + 28.78 * (v**2) - 5603 * v + 4.566 * (10**5)
        else:
            return 0

    def braking(self):
        v = self.v
        if 0 <= v <= 20:
            return 9925 * v + 1.243
        if 20 <= v <= 100:
            return 2.039 * (10**-13) * v + 1.985 * (10**5)
        if 100 <= v <= 220:
            return 5.389 * (v**2) - 2583 * v + 4.012 * (10**5)
        else:
            return 0

    def acc(self, demand=1):
        v = self.v
        if demand >= 0:
            return self.acc_pos(v, demand)
        else:
            return self.acc_neg(v, demand)

    def acc_pos(self, v, demand=1):
        F = self.traction() * demand
        F_available = F - self.A * (v**2) - self.B * v - self.C
        acc = F_available / self.m
        return acc * 3.6

    def acc_neg(self, v, demand=1):
        F = self.braking() * demand
        F_available = F - self.A * (v**2) - self.B * v - self.C
        acc = F_available / self.m
        return acc * 3.6

    def speed(self, demand=1):
        acc = self.acc(demand)
        self.acc_list.append(acc)
        # Use previous speed to calculate new speed
        old_v = self.v
        new_v = old_v + acc

        # Ensure speed is never negative
        new_v = max(0, new_v)
        # Also cap at maximum speed
        new_v = min(new_v, self.v_max)

        # If we're at zero speed and trying to accelerate backward, force demand to zero
        # to prevent the train from getting stuck at zero speed
        if old_v == 0 and acc < 0:
            acc = 0
            new_v = 0

        self.v = new_v
        self.v_list.append(self.v)

    def cost(self, demand=1):
        F = demand
        F_max = 1
        V = self.v
        V_max = self.v_max
        V_safe = min(V, V_max)
        # Ensure V is within a safe range for math.exp
        c = (
            (1 - math.exp(-V_safe / (V_max + 0.001)))
            * math.tanh(100 * F)
            * (1 - math.exp(-abs(F) / (F_max + 0.001)))
        )
        Puti = (c / self.c_max) * self.p_max
        self.power_usage += Puti
        self.power_usage_list.append(Puti)

    def iterate(self, demand):
        vi = self.v
        self.demand_list.append(demand)
        self.speed(demand)
        self.cost(demand)
        # Calculate distance increment using trapezoidal rule
        self.distance += vi + (self.v - vi) / 2
        self.distance_list.append(self.distance)

    def reset(self):
        self.v = 0
        self.v_list = [0]
        self.demand_list = []
        self.power_usage = 0
        self.distance = 0
        self.distance_list = [0]
        self.power_usage_list = []
        self.acc_list = []

    def simulate_complete_journey(self, demand_list):
        self.reset()
        for demand in demand_list:
            self.iterate(demand)
        return {
            "distance": self.distance,
            "final_speed": self.v,
            "power_usage": self.power_usage,
            "speeds": self.v_list,
            "distances": self.distance_list,
            "accelerations": self.acc_list,
            "power_usages": self.power_usage_list,
        }

    def objective(self, demand_list, D, penalty_weight=10000):
        results = self.simulate_complete_journey(demand_list)

        # Penalties
        final_speed_penalty = penalty_weight * abs(results["final_speed"])
        distance_penalty = penalty_weight * abs(results["distance"] - D)

        # Special penalty for zero or very small movement solutions
        zero_movement_penalty = 0
        if results["distance"] < 1.0:  # If distance is less than 1 km
            zero_movement_penalty = penalty_weight * 100  # Very harsh penalty

        # Add penalty for excessive demand changes (to promote smoother control)
        demand_changes = np.abs(np.diff(demand_list))
        smoothness_penalty = np.sum(demand_changes) * penalty_weight * 0.01

        total_objective = (
            results["power_usage"]
            + final_speed_penalty
            + distance_penalty
            + zero_movement_penalty
            + smoothness_penalty
        )

        # Debug output for extremely small distances
        if results["distance"] < 1.0:
            print(
                f"Warning: Near-zero distance solution detected: {results['distance']:.4f} km"
            )
            print(f"Max speed reached: {max(results['speeds']):.2f} km/h")
            # Print the first few demands to diagnose issues
            print(f"First 10 demands: {demand_list[:10]}")

        return total_objective

    def evaluate_constraints(self, demand_list, D, tolerance=0.1):
        results = self.simulate_complete_journey(demand_list)

        distance_error = abs(results["distance"] - D)
        min_speed = min(results["speeds"])
        final_speed = results["final_speed"]

        return {
            "distance": results["distance"],
            "target_distance": D,
            "distance_error": distance_error,
            "min_speed": min_speed,
            "final_speed": final_speed,
            "constraints_satisfied": (
                distance_error < tolerance
                and min_speed >= 0
                and abs(final_speed) < tolerance
            ),
        }

    def distance_constraint(self, demand_list, D, tolerance=0.1):
        results = self.simulate_complete_journey(demand_list)
        return tolerance - abs(results["distance"] - D)

    def min_speed_constraint(self, demand_list):
        results = self.simulate_complete_journey(demand_list)
        min_speed = min(results["speeds"])
        return min_speed

    def final_speed_constraint(self, demand_list, tolerance=0.1):
        results = self.simulate_complete_journey(demand_list)
        return tolerance - abs(results["final_speed"])

    def minimum_distance_constraint(self, demand_list):
        results = self.simulate_complete_journey(demand_list)
        # Return positive value if distance is more than 10% of target
        return results["distance"] - 1.0  # At least 1 km

    def optimize(self, D, num_steps=100, bounds=None, maxiter=10000):
        if bounds is None:
            bounds = [(-1, 1) for _ in range(num_steps)]

        # Create a very specific initial guess that absolutely ensures movement
        initial_demand_list = np.zeros(num_steps)
        accel_phase = int(num_steps * 0.4)  # Longer acceleration phase
        cruise_phase = int(num_steps * 0.4)
        brake_phase = num_steps - accel_phase - cruise_phase

        # Strong initial acceleration
        initial_demand_list[:accel_phase] = 1.0  # Maximum acceleration
        initial_demand_list[accel_phase : accel_phase + cruise_phase] = (
            0.2  # Light positive demand during cruise
        )
        initial_demand_list[accel_phase + cruise_phase :] = -1.0  # Maximum braking

        # Verify the initial guess actually moves the train
        initial_results = self.simulate_complete_journey(initial_demand_list)
        print(
            f"Initial guess results: Distance = {initial_results['distance']:.2f} km, "
            f"Max speed = {max(initial_results['speeds']):.2f} km/h"
        )

        if initial_results["distance"] < 1.0:
            print("WARNING: Initial guess doesn't move the train! Adjusting...")
            # Force stronger acceleration for longer
            initial_demand_list[: int(num_steps * 0.6)] = 1.0
            initial_demand_list[int(num_steps * 0.6) :] = -1.0

            # Verify again
            initial_results = self.simulate_complete_journey(initial_demand_list)
            print(
                f"Adjusted initial guess: Distance = {initial_results['distance']:.2f} km, "
                f"Max speed = {max(initial_results['speeds']):.2f} km/h"
            )

        # Progressive optimization approach
        # First, optimize just to get the train moving to the target distance
        print("\nStage 1: Optimizing to reach target distance...")

        # Initially, just focus on reaching the target distance
        constraints_stage1 = [
            {
                "type": "ineq",
                "fun": self.minimum_distance_constraint,  # Ensure train moves
            },
            {
                "type": "ineq",
                "fun": self.distance_constraint,
                "args": (D, 5.0),  # Looser tolerance initially
            },
        ]

        result_stage1 = minimize(
            lambda x: self.objective(
                x, D, penalty_weight=100
            ),  # Lower penalty weight initially
            initial_demand_list,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_stage1,
            options={"maxiter": maxiter // 2, "disp": True, "ftol": 1e-4},
        )

        # Check if we got a reasonable solution from stage 1
        stage1_demands = result_stage1.x
        stage1_results = self.simulate_complete_journey(stage1_demands)
        print(
            f"Stage 1 results: Distance = {stage1_results['distance']:.2f} km, "
            f"Final speed = {stage1_results['final_speed']:.2f} km/h"
        )

        # Now refine with tighter constraints
        print("\nStage 2: Refining solution with full constraints...")
        constraints_stage2 = [
            {
                "type": "ineq",
                "fun": self.distance_constraint,
                "args": (D, 0.5),  # Tighter tolerance
            },
            {
                "type": "ineq",
                "fun": self.min_speed_constraint,
            },
            {
                "type": "ineq",
                "fun": self.final_speed_constraint,
                "args": (0.5,),  # Slightly relaxed final speed constraint
            },
        ]

        # Use the result from stage 1 as the starting point for stage 2
        result_stage2 = minimize(
            lambda x: self.objective(
                x, D, penalty_weight=5000
            ),  # Higher penalty in stage 2
            stage1_demands,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_stage2,
            options={"maxiter": maxiter // 2, "disp": True, "ftol": 1e-5},
        )

        # Final check
        optimized_demands = result_stage2.x
        constraint_results = self.evaluate_constraints(optimized_demands, D)

        print(
            "\nFinal optimization status:", result_stage2.success, result_stage2.message
        )
        print("Final constraint evaluation:")
        for key, value in constraint_results.items():
            print(f"  {key}: {value}")

        # If the solution is still poor, try one more optimization with very high penalty
        if (
            not constraint_results["constraints_satisfied"]
            and constraint_results["distance"] < D * 0.9
        ):
            print(
                "\nWarning: Poor solution. Attempting final optimization with extreme penalties..."
            )
            result_final = minimize(
                lambda x: self.objective(
                    x, D, penalty_weight=50000
                ),  # Very high penalty
                optimized_demands,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_stage2,
                options={"maxiter": maxiter // 2, "disp": True, "ftol": 1e-6},
            )
            optimized_demands = result_final.x
            constraint_results = self.evaluate_constraints(optimized_demands, D)
            print("Last attempt results:")
            for key, value in constraint_results.items():
                print(f"  {key}: {value}")
            return result_final, constraint_results

        return result_stage2, constraint_results

    def visualize_solution(self, demand_list):
        """
        Create comprehensive visualizations of the optimized solution
        """
        results = self.simulate_complete_journey(demand_list)

        # Create a figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        # Plot distance
        axs[0].plot(results["distances"], "b-")
        axs[0].set_ylabel("Distance (km)")
        axs[0].set_title("Optimized Train Journey")
        axs[0].grid(True)

        # Plot speed
        axs[1].plot(results["speeds"], "r-")
        axs[1].set_ylabel("Speed (km/h)")
        axs[1].grid(True)

        # Plot acceleration
        axs[2].plot(results["accelerations"], "g-")
        axs[2].set_ylabel("Acceleration (km/h/s)")
        axs[2].grid(True)

        # Plot demand
        axs[3].plot(self.demand_list, "k-")
        axs[3].set_ylabel("Demand")
        axs[3].set_xlabel("Time Steps")
        axs[3].grid(True)
        axs[3].set_ylim(-1.1, 1.1)

        plt.tight_layout()
        plt.show()

        # Create another figure for power usage
        plt.figure(figsize=(10, 5))
        plt.plot(results["power_usages"], "b-")
        plt.title("Power Usage")
        plt.xlabel("Time Steps")
        plt.ylabel("Power")
        plt.grid(True)
        plt.show()

        # Print summary statistics
        print(f"Total distance: {results['distance']:.2f} km")
        print(f"Final speed: {results['final_speed']:.2f} km/h")
        print(f"Total power usage: {results['power_usage']:.2f}")
        print(f"Maximum speed: {max(results['speeds']):.2f} km/h")
        print(f"Maximum acceleration: {max(results['accelerations']):.2f} km/h/s")
        print(f"Maximum deceleration: {min(results['accelerations']):.2f} km/h/s")


# Define the problem parameters
A, B, C, m, v_max, p_max = 0.632, 40.7, 3900, 320000, 220, 4305220
problem = Alstom_Problem(A, B, C, m, v_max=v_max, p_max=p_max)

# Define the distance to travel
D = 500  # Target distance in km

# Perform the optimization with fewer control points for better convergence
print("\n=== Starting optimization ===")
result, constraint_eval = problem.optimize(D, num_steps=100)

# Get the optimized demand list
optimized_demand_list = result.x

# Visualize the optimized solution
problem.visualize_solution(optimized_demand_list)

print("\n=== Optimization Summary ===")
print("Optimization successful:", result.success)
print("Minimum power usage:", result.fun)

# Compare with manual control
print("\n=== Comparison with manual control ===")
opt_results = problem.simulate_complete_journey(optimized_demand_list)
print(f"Optimized power usage: {opt_results['power_usage']:.2f}")
