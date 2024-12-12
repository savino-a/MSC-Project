import Classes
from Classes import Locomotive, Wagon, Convoy

# We define a locomotive
BB60000 = Locomotive(10e6, 72, 4)  # W,t
Standard_Wagon = Wagon(16, 82)
Standard_Wagon.load(66)

Convoy1 = Convoy(BB60000.Pnom, BB60000.Mloco, BB60000.Ne)
for i in range(12):
    Convoy1.add_wag(Standard_Wagon)

max_speed_loco = BB60000.calculate_vmax()
print(
    f"The loco alone has a theoretical max speed of: {round(max_speed_loco*3.6,0)} km/h"
)
max_speed_convoy = Convoy1.calculate_vmax()
print(
    f"The loco with the wagons has a theoretical max speed of: {round(max_speed_convoy*3.6,0)} km/h"
)

[Convoy1.delta_vs, Convoy1.speeds] = Convoy1.delta_v_profile(1)
