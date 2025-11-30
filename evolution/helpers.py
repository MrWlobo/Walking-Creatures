import json
from pathlib import Path
from evolution.neural_network import NeuralNetwork


def serialize_network(network: NeuralNetwork, path: Path, filename: str) -> None:
    """
    Save a NeuralNetwork object to a JSON file.

    Args:
        network (NeuralNetwork): The network to serialize.
        path (Path): Path to save the network in (with filename).
        filename (str): Filename to save the network as, without extension.
    """

    data = {
        "nodes": network.nodes,
        "connections": {f"{k[0]},{k[1]}": v for k, v in network.connections.items()}
    }

    file = path / f"{filename}.json"
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "w") as f:
        json.dump(data, f, indent=2)

def deserialize_network(file: Path) -> NeuralNetwork:
    """
    Load a NeuralNetwork object from a JSON file.

    Args:
        file (Path): Path of the JSON file to load.

    Returns:
        NeuralNetwork: A new NeuralNetwork object initialized with the saved nodes and connections.

    The function expects a file named "{filename}.json" containing the 'nodes' and 'connections' keys.
    """

    if not file.exists():
        raise FileNotFoundError(f"There is no file: {file}")

    with open(file, "r") as f:
        data = json.load(f)
    nodes = {int(k): v for k, v in data["nodes"].items()}
    connections = {tuple(map(int, k.split(","))): v for k, v in data["connections"].items()}

    return NeuralNetwork(nodes=nodes, connections=connections)