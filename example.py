import pybullet as p
import pybullet_data
import time
import random
from pathlib import Path
import numpy as np

from simulation.simulation import Simulation

#
#  a temporary test file
#

simulation = Simulation(p.GUI, Path("assets/creatures/6-revolute_biped.urdf"))

simulation.reset_state()

for i in range(1000000):
    simulation.moveRevolute(np.array([0, 1, 2]), np.array([100, 100, 100]))
    simulation.step()
    print(simulation.get_creature_position())
    print(simulation.get_tick_count())
    
    if i % 2000 == 0:
        simulation.reset_state()