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
from core.orchestrate import run_individual, run_population
from evolution.population import generate_random_individual
from evolution.genome_visualization import visualize_neat_network

#
#  a temporary test file
#

if __name__ == "__main__":
    start_time = time.perf_counter()

    pop = [generate_random_individual(6, 0, 1) for _ in range(1000)]
    results = run_population(pop, Path("assets/creatures/1-spherical_hopper.urdf"), FullJointStateGetter(), 1000, TimeOnlyRunConditions(10.0))


    for r in results:
        print(r)
    
    print(len(results))

    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"The code executed in {execution_time:.6f} seconds")