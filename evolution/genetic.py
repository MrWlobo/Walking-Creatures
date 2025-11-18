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

def mutation():
    pass

def selection():
    pass