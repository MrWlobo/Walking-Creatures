import copy
import random
import numpy as np

from core.types import GeneticAlgorithmParams
from evolution.genetic import crossover, mutate
from evolution.neural_network import NeuralNetwork


def generate_random_population(n_individuals: int, input_units: int, units_1d: int, units_3d: int) -> list[NeuralNetwork]:
    """Generate a new population of n_individuals random NEAT-style neural networks.

    Args:
        n_individuals (int): Number of input neurons.
        input_units (int): Number of 1D output neurons.
        units_1d (int): Number of 1D output neurons.
        units_3d (int): Number of 3D output neurons (each counts as 3 neurons).

    Returns:
        list[NeuralNetwork]: A new population of networks with fully connected input-to-output layers,
                                with random weights and random enabled/disabled connections.
    """
    return [generate_random_individual(input_units, units_1d, units_3d) for _ in range(n_individuals)]


def generate_random_individual(input_units: int, units_1d: int, units_3d: int) -> NeuralNetwork:
    """
    Generate a new random NEAT-style neural network.

    Args:
        input_units (int): Number of input neurons.
        units_1d (int): Number of 1D output neurons.
        units_3d (int): Number of 3D output neurons (each counts as 3 neurons).

    Returns:
        NeuralNetwork: A new network with fully connected input-to-output layers,
                        with random weights and random enabled/disabled connections.
    """
    network = NeuralNetwork(input_units, units_1d, units_3d)
    return network


def create_next_generation(population: list[list[NeuralNetwork]], new_species_sizes: list[int], params: GeneticAlgorithmParams) -> list[NeuralNetwork]:
    """Creates a new (not speciated) population from an old (speciated) population.

    Args:
        population (list[list[NeuralNetwork]]): The speciated population the generate the new population from.
        new_species_sizes (list[int]): The respective desired sizes of each species in the population. 
        params (GeneticAlgorithmParams): Parameters of the GA.

    Returns:
        list[NeuralNetwork]: The new population, without defined species.
    """
    if len(population) != len(new_species_sizes):
        raise ValueError(f"Number of species doesn't match number of new species sizes list, population: {population}, species sizes: {new_species_sizes}")
    
    crossover_thresh = params.genetic_operation_ratios[0]
    mutation_thresh = crossover_thresh + params.genetic_operation_ratios[1]
    
    new_population = []
    
    for species, new_size in zip(population, new_species_sizes):
        for _ in range(new_size):
            r = random.random()
            
            offspring = None
            
            if r < crossover_thresh:
                indiv1, indiv2 = params.selection.select(species), params.selection.select(species)
                offspring = crossover(indiv1, indiv2)
            elif r < mutation_thresh:
                indiv = params.selection.select(species)
                offspring = mutate(params.mutation_type_percentages, copy.deepcopy(indiv))
            else:
                indiv = params.selection.select(species)
                offspring = copy.deepcopy(indiv)
            
            if not isinstance(offspring, NeuralNetwork):
                raise RuntimeError(f"Offspring of type {type(offspring)} was generated instead of NeuralNetwork.")
            
            new_population.append(offspring)
    
    if sum([len(species) for species in population]) != len(new_population):
        raise RuntimeError(f"New population size={len(new_population)} is not equal to the old population size={sum([len(species) for species in population])}")
    
    return new_population


def create_species(population: list[NeuralNetwork], coefficients: tuple[float, float, float], compatibility_distance: float) -> list[list[NeuralNetwork]]:
    """
    Divide a population of neural networks into species based on their genetic similarity.

    Args:
        population (list[NeuralNetwork]): List of neural networks to speciate.
        coefficients (tuple[float, float, float]): Weights for (excess, disjoint, weight differences) in compatibility calculation.
        compatibility_distance (float): Threshold for determining whether two networks belong to the same species.

    Returns:
        list[list[NeuralNetwork]]: A list of species, each of which is a list of neural networks.
    """

    if len(coefficients) != 3:
        raise ValueError("There should be exactly 3 coefficients.")

    if not population:
        raise ValueError("Population cannot be empty.")

    speciated_population = [[population[0]]]

    for individual in population[1:]:
        added_to_species = False
        for species in speciated_population:
            if _individuals_within_compatibility_distance(individual, random.choice(species), coefficients, compatibility_distance):
                species.append(individual)
                added_to_species = True
                break

        if not added_to_species:
            speciated_population.append([individual])

    return speciated_population


def _individuals_within_compatibility_distance(network1: NeuralNetwork, network2: NeuralNetwork, coefficients: tuple[float, float, float], compatibility_distance: float) -> bool:
    """
    Determine whether two neural networks are within a specified compatibility distance.

    Args:
        network1 (NeuralNetwork): The first neural network.
        network2 (NeuralNetwork): The second neural network.
        coefficients (tuple[float, float, float]): Weights for (excess, disjoint, weight differences) in the distance calculation.
        compatibility_distance (float): Threshold distance below which networks are considered compatible (same species).

    Returns:
        bool: True if the networks are compatible (distance < compatibility_distance), False otherwise.
    """

    if len(coefficients) != 3:
        raise ValueError("There should be exactly 3 coefficients.")

    c1, c2, c3 = coefficients

    excess_count = 0
    disjoint_count = 0
    weight_differences = []
    normalization_value = max(len(network1.connections), len(network2.connections), 1)

    for connection in network1.connections:
        if connection not in network2.connections and (connection[0] not in network2.nodes or connection[1] not in network2.nodes):
            excess_count += 1
        elif connection not in network2.connections:
            disjoint_count += 1
        else:
            weight_differences.append(abs(network1.connections[connection]["weight"] - network2.connections[connection]["weight"]))

    average_weight_differences = sum(weight_differences) / len(weight_differences) if weight_differences else 0

    return (c1 * excess_count) / normalization_value + (c2 * disjoint_count) / normalization_value + c3 * average_weight_differences < compatibility_distance


def calculate_new_species_sizes(species: list[list[NeuralNetwork]]) -> list[int]:
    """
    Calculate the new number of individuals for each species based on adjusted fitness.
    Each species' size is determined relative to the population mean adjusted fitness:
    - Species with above-average fitness grow.
    - Species with below-average fitness shrink.
    - The total number of individuals across all species remains constant.
    
    Uses proportional allocation to ensure stability.
    Args:
        species (list[list[NeuralNetwork]]): A list of species, where each species is a list of NeuralNetwork instances.
            Each NeuralNetwork is expected to have a 'fitness_value' attribute.
    Returns:
        list[int]: A list of integers representing the new sizes of each species.
            The sum of all integers equals the total population size.
    Raises:
        ValueError: If the input list of species is empty.
    """
    if not species:
        raise ValueError("Number of species cannot be 0.")

    population_size = sum(len(s) for s in species)

    adjusted_fitness_values = [[individual.fitness_value / len(spc) for individual in spc] for spc in species]

    total_adj_fitness = sum(sum(values) for values in adjusted_fitness_values)

    new_sizes = []

    if total_adj_fitness == 0:
        base_size = population_size // len(species)
        new_sizes = [base_size] * len(species)
    else:
        for values in adjusted_fitness_values:
            species_sum = sum(values)
            size = int((species_sum / total_adj_fitness) * population_size)
            new_sizes.append(size)

    diff = population_size - sum(new_sizes)
    
    if diff != 0:
        largest_species_index = new_sizes.index(max(new_sizes))
        new_sizes[largest_species_index] += diff

    return new_sizes