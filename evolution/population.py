import copy
import heapq
import multiprocessing
import os
import random
import numpy as np

from core.types import GeneticAlgorithmParams
from evolution.genetic import crossover, mutate
from evolution.neural_network import NeuralNetwork


def generate_random_population(n_individuals: int, input_units: int, units_1d: int, units_3d: int, initial_connections: int) -> list[NeuralNetwork]:
    """Generate a new population of n_individuals random NEAT-style neural networks.

    Args:
        n_individuals (int): Number of input neurons.
        input_units (int): Number of 1D output neurons.
        units_1d (int): Number of 1D output neurons.
        units_3d (int): Number of 3D output neurons (each counts as 3 neurons).
        initial_connections (int): Number of initial random connections in each NN.

    Returns:
        list[NeuralNetwork]: A new population of networks with fully connected input-to-output layers,
                                with random weights and random enabled/disabled connections.
    """
    return [generate_random_individual(input_units, units_1d, units_3d, initial_connections) for _ in range(n_individuals)]


def generate_random_individual(input_units: int, units_1d: int, units_3d: int, initial_connections: int) -> NeuralNetwork:
    """
    Generate a new random NEAT-style neural network.

    Args:
        input_units (int): Number of input neurons.
        units_1d (int): Number of 1D output neurons.
        units_3d (int): Number of 3D output neurons (each counts as 3 neurons).
        initial_connections (int): Number of initial random connections in NN.

    Returns:
        NeuralNetwork: A new network with fully connected input-to-output layers,
                        with random weights and random enabled/disabled connections.
    """
    network = NeuralNetwork(input_units, units_1d, units_3d, beginning_connections=initial_connections)
    return network


def create_next_generation(population: list[list[NeuralNetwork]], new_species_sizes: list[int], params: GeneticAlgorithmParams) -> list[NeuralNetwork]:
    """Creates a new (not speciated) population from an old (speciated) population.
    
    The process is parallelized, using batches to minimize communication between threads.
    """
    
    if len(population) != len(new_species_sizes):
        raise ValueError(f"Number of species ({len(population)}) doesn't match new species sizes ({len(new_species_sizes)}).")
        
    if params.n_processes is None:
        num_processes = os.cpu_count() or 1
    else:
        num_processes = params.n_processes
    
    total_target_size = sum(new_species_sizes)
    
    # global elitism
    all_individuals = [indiv for species in population for indiv in species]
    
    n_global_elites = int(np.ceil(params.succession_ratio * total_target_size))
    n_global_elites = min(n_global_elites, total_target_size)
    
    elite_indivs = heapq.nlargest(n_global_elites, all_individuals, key=lambda i: i.fitness_info)
    
    new_population = [copy.deepcopy(i) for i in elite_indivs]
    
    # calculate number of remaining spots in the new population (after elitism)
    remaining_spots = total_target_size - len(new_population)
    
    if remaining_spots > 0:
        tasks = []
        
        offspring_counts = []
        total_planned_size = sum(new_species_sizes)
        
        if total_planned_size > 0:
            floored_counts = [int(remaining_spots * (size / total_planned_size)) for size in new_species_sizes]
            
            remainder = remaining_spots - sum(floored_counts)
            
            if remainder > 0:
                sorted_indices = np.argsort(new_species_sizes)[::-1]
                for i in range(remainder):
                    target_idx = sorted_indices[i % len(sorted_indices)]
                    floored_counts[target_idx] += 1
            
            offspring_counts = floored_counts
        else:
            # edge case: if desired size is equal to 0
            offspring_counts = [0] * len(new_species_sizes)

        with multiprocessing.Pool(processes=num_processes) as pool:
            for species, count in zip(population, offspring_counts):
                if count <= 0 or len(species) == 0:
                    continue
                
                
                base_chunk = count // num_processes
                remainder_chunk = count % num_processes
                
                # for small species using multiple threads is unnecessary
                if count < num_processes:
                    tasks.append((species, count, params))
                else:
                    for i in range(num_processes):
                        chunk = base_chunk + (1 if i < remainder_chunk else 0)
                        if chunk > 0:
                            tasks.append((species, chunk, params))

            # offspring generation
            if tasks:
                results = pool.map(_generate_offspring_batch, tasks)
                for batch in results:
                    new_population.extend(batch)

    # final check
    if len(new_population) != total_target_size:
        raise RuntimeError(f"Generated {len(new_population)}, expected {total_target_size}")
        
    return new_population


def _generate_offspring_batch(args):
    """
    Worker: Generates 'batch_size' offspring for the provided species.
    """
    species, batch_size, params = args
    
    if batch_size <= 0:
        return []

    random.seed() 
    np.random.seed()

    batch_results = []
    crossover_thresh = params.genetic_operation_ratios[0]
    
    for _ in range(batch_size):
        offspring = None
        
        # if species is empty or too small for crossover
        if not species:
            offspring = generate_random_individual(
                params.input_units, params.units_1d, params.units_3d, params.initial_connections
            )
        else:
            r = random.random()
            
            # perform crossover if rolled and we have at least 2 parents to cross
            if r < crossover_thresh and len(species) > 1:
                indiv1 = params.selection.select(species)
                indiv2 = params.selection.select(species)
                offspring = crossover(indiv1, indiv2)
            else:
                # mutate otherwise
                indiv = params.selection.select(species)
                offspring = mutate(params.mutation_type_percentages, copy.deepcopy(indiv), params.weight_mutation_params)
        
        if not isinstance(offspring, NeuralNetwork):
            raise RuntimeError(f"Worker generated {type(offspring)} instead of NeuralNetwork")
        
        # add this species' new members to the reults
        batch_results.append(offspring)
        
    return batch_results


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

    min_fitness = min(ind.fitness_value for s in species for ind in s)
    
    offset = 0
    if min_fitness < 0:
        offset = abs(min_fitness)

    adjusted_fitness_values = []
    for spc in species:
        adj_values = [(ind.fitness_value + offset) / len(spc) for ind in spc]
        adjusted_fitness_values.append(adj_values)

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