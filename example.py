import pybullet as p
import pybullet_data
import time
import random
from pathlib import Path
import numpy as np

from simulation.simulation import Simulation
from core.data_utils import FullJointStateGetter
from evolution.neural_network import NeuralNetwork
from core.types import RunResult, TimeOnlyRunConditions
from core.orchestrate import run_individual
from evolution.population import generate_random_individual
from evolution.genome_visualization import visualize_neat_network

#
#  a temporary test file
#

simulation = Simulation(p.DIRECT, Path("assets/creatures/1-spherical_hopper.urdf"))
nn = generate_random_individual(6, 0 ,1)


result = run_individual(nn, simulation, FullJointStateGetter(), 1000, TimeOnlyRunConditions(10.0))

print(result)
