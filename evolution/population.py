from evolution.genetic import mutate
from evolution.neural_network import NeuralNetwork


def generate_random_individual(input_units: int, units_1d: int, units_3d: int, hidden: int = 0) -> NeuralNetwork:
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
    for _ in range(hidden):
        mutate([0, 0, 100], network)
    return network