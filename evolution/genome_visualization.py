import networkx as nx
import matplotlib.pyplot as plt

from evolution.genetic import mutation
from evolution.neural_network import NeuralNetwork

def visualize_neat_network(nn: NeuralNetwork) -> None:
    """
    Visualize a NEAT-style neural network with input nodes on the left,
    output nodes on the right, and hidden nodes in between.

    Args:
        nn (NeuralNetwork): Your neural network instance.
    """
    G = nx.DiGraph()

    for node_id, node_type in nn.nodes.items():
        G.add_node(node_id, label=node_type)

    for (src, tgt), conn in nn.connections.items():
        if conn["enabled"]:
            G.add_edge(src, tgt, weight=conn["weight"])

    layers = {"input": [], "hidden": [], "output": []}
    for node_id, node_type in nn.nodes.items():
        layers[node_type].append(node_id)

    pos = {}
    for i, node_id in enumerate(sorted(layers["input"])):
        pos[node_id] = (0, -i)
    for i, node_id in enumerate(sorted(layers["hidden"])):
        pos[node_id] = (1, -i)
    for i, node_id in enumerate(sorted(layers["output"])):
        pos[node_id] = (2, -i)

    node_colors = []
    for node_id in G.nodes:
        node_type = nn.nodes[node_id]
        if node_type == "input":
            node_colors.append("skyblue")
        elif node_type == "output":
            node_colors.append("lightgreen")
        else:
            node_colors.append("orange")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15)
    nx.draw_networkx_labels(G, pos, {n: f"{n}\n{nn.nodes[n]}" for n in G.nodes}, font_size=8)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.axis('off')
    plt.show()

# Tests
nn = NeuralNetwork(input_units=3, units_1d=2, units_3d=1)
mutation([30, 70, 0], nn)
mutation([10, 80, 10], nn)
mutation([0, 50, 50], nn)
visualize_neat_network(nn)