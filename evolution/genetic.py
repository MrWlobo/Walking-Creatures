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

    # ensure the child's id is the same as the fitter parent's id
    child = NeuralNetwork(nodes=new_nodes, connections=new_connections, id=fitter.id)
    return child


def mutate(probabilities: list[int], individual: NeuralNetwork, weight_mutation_params: tuple[float, float, float, float, float] = (0.8, -2.2, 2.2, -10.0, 10.0)) -> NeuralNetwork:
    """
    Perform a mutation on an individual neural network based on given probabilities.

    Args:
        probabilities (list[int]): List of 3 integers representing percent chance for:
            [0] mutate weight, [1] mutate connection, [2] mutate node.
            Must sum to 100.

        individual (NeuralNetwork): The neural network to mutate.

        weight_mutation_params (tuple[float, float, float, float, float], optional):
            Parameters controlling weight mutation:
            (
                perturbation_chance,  # Probability of perturbing vs replacing a weight
                perturbation_min,     # Minimum additive perturbation
                perturbation_max,     # Maximum additive perturbation
                change_min,           # Minimum value for full weight replacement
                change_max            # Maximum value for full weight replacement
            )
            Defaults to (0.8, -2.2, 2.2, -10.0, 10.0).
    """

    if len(probabilities) != 3:
        raise ValueError("There should be exactly 3 values in probabilities.")

    if sum(probabilities) != 100:
        raise ValueError("Probabilities should sum up to 100.")

    selected = random.uniform(1, 100)

    if selected <= probabilities[0]:
        connection = random.choice(list(individual.connections.keys()))
        _mutate_weight(connection, individual, weight_mutation_params)

    elif selected <= (probabilities[0] + probabilities[1]):
        success = False
        possible_pairs = [(a, b) for a in individual.nodes for b in individual.nodes if a != b]
        random.shuffle(possible_pairs)
        
        while possible_pairs and not success:
            connection = possible_pairs.pop()
            success = _mutate_connection(connection, individual)

    else:
        success = False
        while not success:
            connection = random.choice(list(individual.connections.keys()))
            success = _mutate_node(connection, individual)
    
    # reset the id after mutation to ensure the individual gets a new random seed
    individual.reset_id()

    return individual


def _mutate_weight(connection: tuple[int, int], individual: NeuralNetwork, weight_mutation_params: tuple[float, float, float, float, float]) -> None:
    """
    Mutate the weight of a given connection in a neural network.

    Args:
        connection (tuple[int, int]): The (source, target) node indices of the connection.

        individual (NeuralNetwork): Neural network to mutate.

        weight_mutation_params (tuple[float, float, float, float, float], optional):
            Parameters controlling weight mutation:
            (
                perturbation_chance,  # Probability of perturbing vs replacing a weight
                perturbation_min,     # Minimum additive perturbation
                perturbation_max,     # Maximum additive perturbation
                change_min,           # Minimum value for full weight replacement
                change_max            # Maximum value for full weight replacement
            )
    """

    perturbation_chance, perturbation_min, perturbation_max, change_min, change_max = weight_mutation_params

    if perturbation_chance < 0 or perturbation_chance > 1:
        raise ValueError("Perturbation chance (weight_mutation_params[0]) should have a value between 0 and 1.")

    if perturbation_min > perturbation_max:
        raise ValueError("Minimum perturbation value (weight_mutation_params[1]) cannot be bigger than maximum perturbation value (weight_mutation_params[2]).")

    if change_min > change_max:
        raise ValueError("Minimum change value (weight_mutation_params[3]) cannot be bigger than maximum change value (weight_mutation_params[4]).")

    if random.random() < perturbation_chance:
        individual.connections[connection]["weight"] += random.uniform(perturbation_min, perturbation_max)
    else:
        individual.connections[connection]["weight"] = random.uniform(change_min, change_max)


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