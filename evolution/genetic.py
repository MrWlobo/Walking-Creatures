import random
import copy

from evolution.fitness import XDistanceFitness
from evolution.neural_network import NeuralNetwork

def generate_random(input_units, units_1d, units_3d) -> NeuralNetwork:
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

    return NeuralNetwork(input_units, units_1d, units_3d)

def crossover(parent1: NeuralNetwork, parent2: NeuralNetwork, p1_fitness: float, p2_fitness: float) -> NeuralNetwork:
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

    fitter = parent1 if p1_fitness > p2_fitness else parent2
    other = parent1 if p1_fitness <= p2_fitness else parent2

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

def mutation(probabilities: list[int], individual: NeuralNetwork):

    if len(probabilities) != 3:
        raise ValueError("There should be exactly 3 values in probabilities.")

    if sum(probabilities) != 100:
        raise ValueError("Probabilities should sum up to 100.")

    selected = random.uniform(0, 100)

    if selected <= probabilities[0]:
        connection = random.choice(list(individual.connections.keys()))
        mutate_weight(connection, individual)

    elif selected <= (probabilities[0] + probabilities[1]):
        success = False
        while not success:
            node_pairs = [(a, b) for a in individual.nodes for b in individual.nodes
                          if a != b]
            if not node_pairs:
                break
            connection = random.choice(node_pairs)
            success = mutate_connection(connection, individual)

    else:
        success = False
        while not success:
            connection = random.choice(list(individual.connections.keys()))
            success = mutate_node(connection, individual)

def mutate_weight(connection: tuple[int, int], individual: NeuralNetwork):
    if random.random() < 0.9:
        individual.connections[connection]["weight"] += random.uniform(-0.1, 0.1)
    else:
        individual.connections[connection]["weight"] = random.uniform(-1.0, 1.0)

def mutate_connection(connection: tuple[int, int], individual: NeuralNetwork) -> bool:
    mutated = True
    if connection not in individual.connections:
        mutated = individual.add_connection(connection)
    else:
        individual.toggle_connection(connection)

    return mutated

def mutate_node(connection: tuple[int, int], individual: NeuralNetwork):
    mutated = True
    if connection in individual.connections:
        mutated = individual.add_connection(connection)

    return mutated

def selection():
    pass