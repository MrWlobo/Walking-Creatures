import math
import dill
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pybullet as p
from collections.abc import Callable

from core.types import GeneticAlgorithmParams
from evolution.fitness import FitnessStats, evaluate_population
from evolution.helpers import serialize_network
from evolution.neural_network import NeuralNetwork
from evolution.population import calculate_new_species_sizes, create_next_generation, create_species, generate_random_population
from evolution.visualization import Visualization
from simulation.simulation import Simulation

matplotlib.use('Agg')

class GeneticAlgorithm:
    """A Genetic Algorithm class.
    1. Initialize with a GeneticAlgorithmParams object.
    2. Call evolve().
    """
    def __init__(self, params: GeneticAlgorithmParams):
        self.params: GeneticAlgorithmParams = params
        self.best_fitness_values = []
        self.mean_fitness_values = []
        
        # check if paths are relative, to make saved results retrievable
        if params.results_path.is_absolute():
            raise ValueError(f"Results path should be a relative path, got: {params.results_path}")
        
        if params.creature_path.is_absolute():
            raise ValueError(f"Creature path should be a relative path, got: {params.creature_path}")
        
        self.project_path = Path(__file__).parent.parent.resolve()
        
        self.results_path = self.project_path / params.results_path
        self.creature_path = self.project_path / params.creature_path
        
        self._check_params_validity()
    
        # calculate NN input and output dimensions
        temp_sim = Simulation(p.DIRECT, self.creature_path)

        self.units_1d: int = temp_sim.num_revolute # 2 values for each 1d joint
        self.units_3d: int = temp_sim.num_spherical # 2 3d vectors for each 3d joint

        self.input_units: int = len(self.params.state_getter.get_state(temp_sim))

        temp_sim.terminate()

        # set the initial population
        self.population: list[NeuralNetwork] | list[list[NeuralNetwork]] = generate_random_population(
            self.params.population_size, 
            self.input_units, 
            self.units_1d, 
            self.units_3d,
            self.params.initial_connections,
        )
        
        # create dir to store GA run results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f"run_{timestamp}"
        self.save_dir: Path = self.results_path / dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # save GA params
        with open((self.save_dir / "params.json").resolve(), "w") as f:
            json.dump(asdict(self.params), f, indent=4, default=str)
        
        with open((self.save_dir / "params.pkl").resolve(), "wb") as f:
            dill.dump(self.params, f)
        
        self.global_best = NeuralNetwork(self.input_units, self.units_1d, self.units_3d)
        self.global_best.fitness_info = float('-inf')
    

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
            self.population = create_next_generation(self.population, new_species_sizes, self.params, generation)
            
            # save generation stats
            self._save_generation_stats(generation, curr_species, initial_fitness_stats, species_fitness_stats)
            
            logging.basicConfig(
                level=logging.INFO,
                format='\033[92m%(levelname)s: %(message)s\033[0m',
                force=True 
            )
            logging.info(f"Generation {generation} finished.")


    def _save_generation_stats(self, generation: int, curr_species: list[list[NeuralNetwork]], initial_fitness_stats: FitnessStats, species_fitness_stats: list[FitnessStats], fitness_plot_interval: int = 20):
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
        
        # save population and fittest individual plots
        viz.visualize_population_with_fittest_individual(curr_species, generation, "population_with_fittest_indiv_img", save_image=True, show_image=False)
        
        # save fittest individual
        best_indiv = max([max(species, key=lambda i: i.fitness_info) for species in curr_species], key=lambda i: i.fitness_info)
        serialize_network(best_indiv, GEN_DIR, "generation_fittest_individual")
        
        # save individual best across all generation
        self.global_best = max([self.global_best, best_indiv], key=lambda i: i.fitness_info)
        serialize_network(self.global_best, GEN_DIR, "global_fittest_individual")
        viz.visualize_network(self.global_best, ax, "global_fittest_individual_img", save_image=True, show_image=False, title=f"Global fittest individual with a fitness value of {self.global_best.fitness_info}")

        self.best_fitness_values.append(initial_fitness_stats.best_fitness)
        self.mean_fitness_values.append(initial_fitness_stats.mean_fitness)

        # save average and best fitness for all generations
        if generation >= fitness_plot_interval and generation % fitness_plot_interval == 0:
            fig, ax = plt.subplots(1, 1)
            viz.line_plot(self.best_fitness_values, ax, "Best fitness across generations", "Best fitness", f"Generation_{generation}_best_fitness_values", True, False)
            fig, ax = plt.subplots(1, 1)
            viz.line_plot(self.mean_fitness_values, ax, "Mean fitness across generations", "Mean fitness",f"Generation_{generation}_mean_fitness_values", True, False)
        
        plt.close('all')
    
    
    def _check_params_validity(self):
        p = self.params
        
        if not self.creature_path.exists():
            raise FileNotFoundError(f"The creature file was not found at: {self.creature_path}")
        
        if self.creature_path.suffix != '.urdf':
            raise ValueError(f"Creature file must be a .urdf file. Got: {self.creature_path.suffix}")
        
        if p.surface_friction < 0 or p.surface_friction > 1.5:
            raise ValueError(f"Surface friction must be in the range (0, 1.5). Got: {p.surface_friction}")

        if any(x is None for x in [p.fitness, p.selection, p.state_getter, p.run_conditions]):
            raise ValueError("fitness, selection, state_getter, and run_conditions must be initialized objects, not None.")

        if p.population_size < 2:
            raise ValueError(f"Population size must be at least 2 to allow for crossover. Got: {p.population_size}")

        if p.n_generations < 1:
            raise ValueError(f"Number of generations must be at least 1. Got: {p.n_generations}")
        
        if p.initial_connections < 0:
            raise ValueError(f"Initial connections cannot be negative. Got: {p.initial_connections}")

        if isinstance(p.succession_ratio, int):
            p.succession_ratio = float(p.succession_ratio)

        if not isinstance(p.succession_ratio, float) and not isinstance(p.succession_ratio, Callable):
            raise ValueError(f"succession_ratio must be a float or Callable Got: {type(p.succession_ratio)}")

        if isinstance(p.succession_ratio, float) and not (0.0 <= p.succession_ratio <= 1.0):
            raise ValueError(f"succession_ratio must be between 0.0 and 1.0. Got: {p.succession_ratio}")

        if not isinstance(p.genetic_operation_ratios, tuple) and not isinstance(p.genetic_operation_ratios, Callable):
            raise ValueError(f"genetic_operation_ratios must be a tuple or Callable Got: {type(p.genetic_operation_ratios)}")

        if isinstance(p.genetic_operation_ratios, tuple) and len(p.genetic_operation_ratios) != 2:
            raise ValueError("genetic_operation_ratios must be a tuple of length 2 (crossover, mutation).")
        
        if isinstance(p.genetic_operation_ratios, tuple) and not all(0.0 <= r <= 1.0 for r in p.genetic_operation_ratios):
            raise ValueError("All values in genetic_operation_ratios must be between 0.0 and 1.0.")

        if not isinstance(p.mutation_type_percentages, list) and not isinstance(p.mutation_type_percentages, Callable):
            raise ValueError(f"mutation_type_percentages must be a tuple or Callable Got: {type(p.mutation_type_percentages)}")

        if isinstance(p.mutation_type_percentages, list) and len(p.mutation_type_percentages) != 3:
            raise ValueError("mutation_type_percentages must have exactly 3 values [weight, connection, node].")
        
        if isinstance(p.mutation_type_percentages, list) and not math.isclose(sum(p.mutation_type_percentages), 100.0, rel_tol=1e-5):
            raise ValueError(f"mutation_type_percentages must sum to 100. Got sum: {sum(p.mutation_type_percentages)}")
        
        if len(p.weight_mutation_params) != 5:
            raise ValueError("weight_mutation_params must have exactly 5 values.")
        
        if p.weight_mutation_params[0] < 0 or p.weight_mutation_params[0] > 1:
            raise ValueError("weight_mutation_params[0] must represent a valid probability value")
        
        if isinstance(p.mutation_after_crossover_probability, int):
            p.mutation_after_crossover_probability = float(p.mutation_after_crossover_probability)

        if not isinstance(p.mutation_after_crossover_probability, float) and not isinstance(p.mutation_after_crossover_probability, Callable):
            raise ValueError(f"mutation_after_crossover_probability must be a float or Callable Got: {type(p.mutation_after_crossover_probability)}")
        
        if isinstance(p.mutation_after_crossover_probability, float) and not (0.0 <= p.mutation_after_crossover_probability <= 1.0):
            raise ValueError(f"mutation_after_crossover_probability must be between 0.0 and 1.0. Got: {p.mutation_after_crossover_probability}")

        if len(p.speciation_coefficients) != 3:
            raise ValueError("speciation_coefficients must have exactly 3 values.")
        
        if p.speciation_compatibility_distance < 0:
            raise ValueError("speciation_compatibility_distance must be non-negative.")

        if p.n_processes is not None and p.n_processes < 1:
            raise ValueError(f"n_processes must be None (auto) or a positive integer. Got: {p.n_processes}")
        
        if p.time_step <= 0:
            raise ValueError("time_step must be positive.")
