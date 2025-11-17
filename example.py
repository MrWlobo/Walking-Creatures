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

simulation = Simulation(p.GUI, Path("assets/creatures/1-spherical_hopper.urdf"))

# simulation.reset_state()

for i in range(1000000):
    # simulation.moveRevolute(np.array([1]), np.array([80]))
    # simulation.moveSpherical(np.array([0]), np.array([[50, 0, 0]]))
    simulation.step()
    # print(simulation.get_base_state())
    # print(simulation.get_tick_count())
    # print(simulation.get_revolute_joint_states())
    print(simulation.get_spherical_joint_states())
    
    # if i % 2000 == 0:
    #     simulation.reset_state()