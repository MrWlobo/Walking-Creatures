import pybullet as p

from core.types import GeneticAlgorithmParams
from evolution.fitness import evaluate_population
from evolution.neural_network import NeuralNetwork
from evolution.population import create_species, generate_random_population
from simulation.simulation import Simulation


class GeneticAlgorithm:
    def __init__(self, params: GeneticAlgorithmParams):
        self.params: GeneticAlgorithmParams = params
    
        # calculate NN input and output dimensions
        temp_sim = Simulation(p.DIRECT, self.params.creature_path)

        self.units_1d: int = temp_sim.num_revolute * 2 # 2 values for each 1d joint
        self.units_3d: int = temp_sim.num_spherical * 6 # 2 3d vectors for each 3d joint

        self.input_units: int = len(self.params.state_getter.get_state(temp_sim))

        temp_sim.terminate()

        # set the initial population
        self.population: list[NeuralNetwork] | list[list[NeuralNetwork]] = generate_random_population(
            self.params.population_size, 
            self.input_units, 
            self.units_1d, 
            self.units_3d
        )
    

    def evolve(self):
        for generation in range(self.params.n_generations):
            self.population = evaluate_population(self.population, self.params)

            # calculate fitness stats for the whole population before speciation
            intiial_fitness_stats = self.params.fitness.getStats(self.population)

            self.population = create_species(self.population, self.params.speciation_coefficients, self.params.speciation_compatibility_distance)


    def _save_generation_stats(self):
        pass

