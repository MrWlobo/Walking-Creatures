import random
import copy
from evolution.neural_network import NeuralNetwork


def crossover(parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
    """
    Perform NEAT-style crossover between two parent networks to produce a child network.

    Matching connections are randomly inherited from either parent.
    Disjoint and excess connections are inherited from the fitter parent.

    Args:
        parent1 (NeuralNetwork): First parent network.
        parent2 (NeuralNetwork): Second parent network.
        p1_fitness (float): Fitness score of parent1.
        p2_fitness (float): Fitness score of parent2.

    Returns:
        NeuralNetwork: Child network created from the crossover of the two parents.
    """

    fitter = parent1 if parent1.fitness_value > parent2.fitness_value else parent2
    other = parent1 if parent1.fitness_value <= parent2.fitness_value else parent2

    new_nodes = copy.deepcopy(fitter.nodes)
    new_connections = {}

    for connection in fitter.connections:
        if connection in other.connections:
            chosen_connection = random.choice([fitter.connections[connection], other.connections[connection]])
        else:
            chosen_connection = fitter.connections[connection]

        new_connections[connection] = copy.deepcopy(chosen_connection)

    child = NeuralNetwork(nodes=new_nodes, connections=new_connections)
    return child

def mutate(probabilities: list[int], individual: NeuralNetwork) -> None:
    """
    Perform a mutation on an individual neural network based on given probabilities.

    Args:
        probabilities (list[int]): List of 3 integers representing percent chance for:
            [0] mutate weight, [1] mutate connection, [2] mutate node.
            Must sum to 100.
        individual (NeuralNetwork): The neural network to mutate.
    """

    if len(probabilities) != 3:
        raise ValueError("There should be exactly 3 values in probabilities.")

    if sum(probabilities) != 100:
        raise ValueError("Probabilities should sum up to 100.")

    selected = random.uniform(1, 100)

    if selected <= probabilities[0]:
        connection = random.choice(list(individual.connections.keys()))
        _mutate_weight(connection, individual)

    elif selected <= (probabilities[0] + probabilities[1]):
        success = False
        while not success:
            node_pairs = [(a, b) for a in individual.nodes for b in individual.nodes
                        if a != b]
            if not node_pairs:
                break
            connection = random.choice(node_pairs)
            success = _mutate_connection(connection, individual)

    else:
        success = False
        while not success:
            connection = random.choice(list(individual.connections.keys()))
            success = _mutate_node(connection, individual)

def _mutate_weight(connection: tuple[int, int], individual: NeuralNetwork) -> None:
    """
    Mutate the weight of a given connection in a neural network.

    Args:
        connection (tuple[int, int]): The (source, target) node indices of the connection.
        individual (NeuralNetwork): Neural network to mutate.

    Behavior:
        - 90% chance: small perturbation [-0.1, 0.1] added to weight
        - 10% chance: assign completely new random weight in [-1.0, 1.0]
    """
    if random.random() < 0.9:
        individual.connections[connection]["weight"] += random.uniform(-0.1, 0.1)
    else:
        individual.connections[connection]["weight"] = random.uniform(-1.0, 1.0)

def _mutate_connection(connection: tuple[int, int], individual: NeuralNetwork) -> bool:
    """
    Mutate a connection by either adding a new connection or toggling an existing one.

    Args:
        connection (tuple[int, int]): The (source, target) node indices.
        individual (NeuralNetwork): Neural network to mutate.

    Returns:
        bool: True if mutation succeeded (connection added or toggled), False otherwise.
    """

    mutated = True
    if connection not in individual.connections:
        mutated = individual.add_connection(connection)
    else:
        individual.toggle_connection(connection)

    return mutated

def _mutate_node(connection: tuple[int, int], individual: NeuralNetwork) -> bool:
    """
    Mutate a network by splitting an existing connection to add a new hidden node.

    Args:
        connection (tuple[int, int]): The (source, target) node indices of the connection.
        individual (NeuralNetwork): Neural network to mutate.

    Returns:
        bool: True if a new node was successfully added, False otherwise.
    """

    mutated = True
    if connection in individual.connections:
        mutated = individual.add_node_to_connection(connection)

    return mutated

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
