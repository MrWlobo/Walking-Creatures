import pybullet as p

from core.types import GeneticAlgorithmParams
from evolution.population import generate_random_population
from simulation.simulation import Simulation


class GeneticAlgorithm:
    def __init__(self, params: GeneticAlgorithmParams):
        self.params: GeneticAlgorithmParams = params
    
        # calculate NN input and output dimensions
        temp_sim = Simulation(p.DIRECT, self.params.creature_path)

        self.units_1d = temp_sim.num_revolute * 2 # 2 values for each 1d joint
        self.units_3d = temp_sim.num_spherical * 6 # 2 3d vectors for each 3d joint

        self.input_units = len(self.params.state_getter.get_state(temp_sim))

        temp_sim.terminate()

        # set the initial population
        self.population = generate_random_population(self.params.population_size, self.input_units, self.units_1d, self.units_3d)