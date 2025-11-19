import random

from evolution.neural_network import NeuralNetwork


def generate_random_individual(input_units, units_1d, units_3d) -> NeuralNetwork:
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