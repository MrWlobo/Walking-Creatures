import pickle
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib 
from matplotlib import pyplot as plt
import pybullet as p

from core.types import GeneticAlgorithmParams
from evolution.fitness import FitnessStats, evaluate_population
from evolution.helpers import serialize_network
from evolution.neural_network import NeuralNetwork
from evolution.population import calculate_new_species_sizes, create_next_generation, create_species, generate_random_population
from evolution.visualization import Visualization
from simulation.simulation import Simulation

matplotlib.use('Agg')

class GeneticAlgorithm:
    def __init__(self, params: GeneticAlgorithmParams):
        self.params: GeneticAlgorithmParams = params
    
        # calculate NN input and output dimensions
        temp_sim = Simulation(p.DIRECT, self.params.creature_path)

        self.units_1d: int = temp_sim.num_revolute # 2 values for each 1d joint
        self.units_3d: int = temp_sim.num_spherical # 2 3d vectors for each 3d joint

        self.input_units: int = len(self.params.state_getter.get_state(temp_sim))

        temp_sim.terminate()

        # set the initial population
        self.population: list[NeuralNetwork] | list[list[NeuralNetwork]] = generate_random_population(
            self.params.population_size, 
            self.input_units, 
            self.units_1d, 
            self.units_3d
        )
        
        # create dir to store GA run results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f"run_{timestamp}"
        self.save_dir: Path = self.params.results_path / dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # save GA params
        with open((self.save_dir / "params.json").resolve(), "w") as f:
            json.dump(asdict(self.params), f, indent=4, default=str)
        
        with open((self.save_dir / "params.pkl").resolve(), "wb") as f:
            pickle.dump(self.params, f)
    

    def evolve(self):
        for generation in range(self.params.n_generations):
            self.population = evaluate_population(self.population, self.params)

            # calculate fitness stats for the whole population before speciation
            initial_fitness_stats = self.params.fitness.getStats(self.population)

            # divide the population into species
            self.population = create_species(self.population, self.params.speciation_coefficients, self.params.speciation_compatibility_distance)
            
            # save the speciated populaton for visualisation purposes
            curr_species = self.population
            
            # get fitness stats for each species
            species_fitness_stats = [self.params.fitness.getStats(species) for species in self.population]
            
            # adjust each species' fitness values
            self.population = self.params.fitness.adjustSpeciesFitness(self.population)
            
            # calculate new species sizes for the purpose of genetic operations
            new_species_sizes = calculate_new_species_sizes(self.population)
            
            # generate a new population using genetic operations
            self.population = create_next_generation(self.population, new_species_sizes, self.params)
            
            # save generation stats
            self._save_generation_stats(generation, curr_species, initial_fitness_stats, species_fitness_stats)
            
            logging.basicConfig(
                level=logging.INFO,
                format='\033[92m%(levelname)s: %(message)s\033[0m',
                force=True 
            )
            logging.info(f"Generation {generation} finished.")


    def _save_generation_stats(self, generation: int, curr_species: list[list[NeuralNetwork]], initial_fitness_stats: FitnessStats, species_fitness_stats: list[FitnessStats]):
        GEN_DIR = self.save_dir / f"generation-{generation}"
        GEN_DIR.mkdir(parents=True, exist_ok=True)
        
        # save pre-speciation fitness
        with open((GEN_DIR / "pre-speciation_fitness.json").resolve(), "w") as f:
            json.dump(asdict(initial_fitness_stats), f, indent=4)
        
        # save fitness for each species
        with open((GEN_DIR / "species_fitness.json").resolve(), "w") as f:
            json.dump([asdict(species_fitness) for species_fitness in species_fitness_stats], f, indent=4, default=str)
        
        viz = Visualization(GEN_DIR)
        fig, ax = plt.subplots(1, 1)
        
        # save population plots
        viz.visualize_population(curr_species, ax, "population_img", save_image=True, show_image=False)
        viz.visualize_population_with_fittest_individual(curr_species, generation, "populatio_with_fittes_indiv_img", save_image=True, show_image=False)
        
        # save fittest individual and its visualisation
        best_indiv = max([max(species, key=lambda i: i.fitness_value) for species in curr_species], key=lambda i: i.fitness_value)
        serialize_network(best_indiv, GEN_DIR, "best_individual")
        viz.visualize_network(best_indiv, ax, "best_individual_img", save_image=True, show_image=False)
        
        plt.close('all')
