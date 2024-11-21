# In this python file, we will try to model a train (BB60000)

## Importing the necessary libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

folder1_path = os.path.join(os.path.dirname(__file__), "..", "Useful_functions")
sys.path.append(folder1_path)

from Equation_solving import newtons_method


# We do not consider the conditions of the track (inclination, turning etc.)
class Wagon:
    def __init__(self, Me, Max_load):
        self.Me = Me  # tonnes
        self.Mrempli = Me
        self.max_load = Max_load

    def load(self, Mload):
        if self.Mrempli >= (self.max_load + self.Me):
            print("Wagon is already full")
        elif (self.Mrempli + Mload) >= (self.max_load + self.Me):
            print("Load too heavy to load")
        else:
            self.Mrempli += Mload

    def RAV(self, v):  # v in m/s
        v = v * 3.6  # v in km/h
        return (
            12 * self.Mrempli + 0.09 * self.Mrempli * v + 0.0044 * self.Mrempli * (v**2)
        )
        # N

    def Presistance(self, v):  # v in m/s
        return v * self.RAV(v)  # W


class Locomotive:
    def __init__(self, Pnom, Mloco, Ne, efficiency=0.85):
        self.Pnom = Pnom
        self.Mloco = Mloco  # tonnes
        self.Mtot = Mloco
        self.Ne = Ne
        self.efficiency = efficiency

    def RAV(self, v):  # v  in m/s
        v = v * 3.6  # v in km/h
        return 6.5 * self.Mloco + 130 * self.Ne + 0.1 * self.Mloco * v + 0.3 * (v**2)

    def Presistance(self, v):  # v in m/s
        return v * self.RAV(v)

    def Pavailable(self, v):  # v in m/s
        return self.Pnom * self.efficiency - self.Presistance(v)

    def delta_v(self, v1):  # v1 in m/s
        Pav = self.Pavailable(v1)
        delta_t = 1  # 1 sec steps
        v2 = math.sqrt((2 / ((self.Mtot) * 1000)) * Pav * delta_t + v1**2)
        return v2 - v1  # delta_v in m/s^2

    def calculate_vmax(self):
        # At vmax, Pavailable=0
        # We will have to solve ax^3 + bx^2 + cx + d = 0
        a = 0.3
        b = 0.1 * self.Mloco
        c = 6.5 * self.Mloco + 130 * self.Ne
        a = a / 3.6
        b = b / 3.6
        c = c / 3.6
        d = -self.Pnom
        vmax_approx = newtons_method(a, b, c, d, 100)
        if vmax_approx == None:
            print("Error in finding vmax, input value")
        else:
            return vmax_approx / 3.6  # m /s


class Convoy(Locomotive):
    def __init__(self, Pnom_loco, Mloco, Ne, efficiency=0.85):
        super().__init__(Pnom_loco, Mloco, Ne, efficiency)
        self.loco = Locomotive(Pnom_loco, Mloco, Ne, efficiency)
        self.list_wag = []
        self.Mtot_wag = 0

    def add_wag(self, wag):
        self.list_wag.append(wag)
        self.Mtot += wag.Mrempli
        self.Mtot_wag += wag.Mrempli

    def Presistance(self, v):
        sum = super().Presistance(v)
        for wag in self.list_wag:
            sum += wag.Presistance(v)
        return sum

    def calculate_vmax(self):
        a = 0.3 + 0.0044 * len(self.list_wag)
        b = 0.1 * self.Mloco + 0.9 * self.Mtot_wag
        c = 6.5 * self.Mloco + 130 * self.Ne + 12 * self.Mtot_wag
        a = a / 3.6
        b = b / 3.6
        c = c / 3.6
        d = -self.Pnom
        vmax_approx = newtons_method(a, b, c, d, 100)
        if vmax_approx == None:
            print("Error in finding vmax, input value")
        else:
            return vmax_approx / 3.6  # m /s


def acceleration_profile(self, verbose=0):
    speed = np.arange(0, self.calculate_vmax(), 0.5)
    delta_v_loco = []
    delta_v_convoy = []
    Pavailable_loco = []
    Pavailable_convoy = []
    for v in speed:
        delta_v_loco.append(self.loco.delta_v(v))
        delta_v_convoy.append(self.delta_v(v))
        Pavailable_loco.append(self.loco.Pavailable(v))
        Pavailable_convoy.append(self.Pavailable(v))

    if verbose == 1:
        # Plotting
        # Create the folder 'plots' if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(
            speed, delta_v_loco, label=r"$\Delta v_{loco}$", color="blue", linewidth=2
        )
        plt.plot(
            speed,
            delta_v_convoy,
            label=r"$\Delta v_{convoy}$",
            color="orange",
            linewidth=2,
        )

        # Add plot details
        plt.title(r"$\Delta v(v)$", fontsize=16)
        plt.xlabel("Speed (m/s)", fontsize=14)
        plt.ylabel(r"$\Delta v$ (m/s²)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the plot
        output_path = "plots/delta_v_v_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(
            speed,
            Pavailable_loco,
            label=r"$P_{available, loco}$",
            color="green",
            linewidth=2,
        )
        plt.plot(
            speed,
            Pavailable_convoy,
            label=r"$P_{available, convoy}$",
            color="red",
            linewidth=2,
        )

        # Add plot details
        plt.title(r"$P_{available}(v)$", fontsize=16)
        plt.xlabel("Speed (m/s)", fontsize=14)
        plt.ylabel(r"$P_{available}$ (W)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save the plot
        output_path = "plots/Pavailable(v).png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    return [delta_v_convoy, speed]


Convoy.delta_v_profile = acceleration_profile
