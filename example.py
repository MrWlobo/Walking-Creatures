import pybullet as p
import pybullet_data
import time
import random
from pathlib import Path
import numpy as np

from simulation.simulation import Simulation
from evolution.neural_network import NeuralNetwork
from evolution.fitness import XDistanceFitness
from core.types import GeneticAlgorithmParams, FullJointStateGetter, TimeOnlyRunConditions
from core.orchestrate import run_population
from evolution.population import generate_random_individual
from evolution.genome_visualization import visualize_neat_network

#
#  a temporary test file
#

if __name__ == "__main__":
    start_time = time.perf_counter()

    pop = [generate_random_individual(16, 2, 2) for _ in range(1000)]

    params = GeneticAlgorithmParams()
    params.creature_path = Path("assets/creatures/2-spherical_2-revolute_two-arm-biped.urdf")
    params.fitness = XDistanceFitness()
    params.indiv_output_scale = 1000
    params.n_processes = None
    params.state_getter = FullJointStateGetter()
    params.run_conditions = TimeOnlyRunConditions(10)
    results = run_population(pop, params)


    for r in results:
        print(r)
    
    print(len(results))

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"The code executed in {execution_time:.6f} seconds")