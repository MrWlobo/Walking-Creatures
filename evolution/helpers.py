import json
from pathlib import Path

from evolution.genome_visualization import visualize_neat_network
from evolution.neural_network import NeuralNetwork


def serialize_network(network: NeuralNetwork, filename: str) -> None:
    """
    Save a NeuralNetwork object to a JSON file.

    Args:
        network (NeuralNetwork): The network to serialize.
        filename (str): Base filename (without extension) to save the network.

    Creates:
        A JSON file named "{filename}.json" containing the network's nodes and connections.
    """

    data = {
        "nodes": network.nodes,
        "connections": {f"{k[0]},{k[1]}": v for k, v in network.connections.items()}
    }

    file = Path("individuals", f"{filename}.json")
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "w") as f:
        json.dump(data, f, indent=2)

def deserialize_network(filename: str) -> NeuralNetwork:
    """
    Load a NeuralNetwork object from a JSON file.

    Args:
        filename (str): Base filename (without extension) of the JSON file to load.

    Returns:
        NeuralNetwork: A new NeuralNetwork object initialized with the saved nodes and connections.

    The function expects a file named "{filename}.json" containing the 'nodes' and 'connections' keys.
    """

    file = Path("individuals", f"{filename}.json")
    if not file.exists():
        raise FileNotFoundError(f"There is no file: {file}.json")

    with open(file, "r") as f:
        data = json.load(f)
    nodes = {int(k): v for k, v in data["nodes"].items()}
    connections = {tuple(map(int, k.split(","))): v for k, v in data["connections"].items()}

    print(connections)

    return NeuralNetwork(nodes=nodes, connections=connections)

nn = NeuralNetwork(input_units=3, units_1d=2, units_3d=1)
serialize_network(nn, "test_individual")
new_nn = deserialize_network("test_individual")
visualize_neat_network(new_nn)