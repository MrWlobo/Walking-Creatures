from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axis import Axis

from evolution.neural_network import NeuralNetwork

class Visualization:

    def __init__(self, save_folder: str):
        self.save_folder = save_folder

    def visualize_network(self, network: NeuralNetwork, axis: Axis, filename: str, save_image: bool = False, show_image: bool = True) -> None:
        """
        Visualize a NEAT-style neural network using NetworkX and Matplotlib.

        The network is drawn with:
        - Input nodes aligned on the left,
        - Hidden nodes in the center,
        - Output nodes on the right,
        with directed edges representing enabled connections.

        Each node is color-coded by type:
            input  - sky blue
            hidden - orange
            output - light green

        Edge labels display connection weights.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network object to visualize. It must provide:
                - `network.nodes`: dict mapping node_id → node_type
                  where node_type ∈ {"input", "hidden", "output"}.
                - `network.connections`: dict mapping (src, tgt) → {"enabled": bool, "weight": float}

        axis : matplotlib.axes.Axes
            The axes to draw the network on.

        filename : str
            The base filename used when saving the visualization image.
            The file will be saved to `visualizations/<filename>.png`.

        save_image : bool, optional (default=False)
            If True, the image is saved using `plt.savefig()`.

        show_image : bool, optional (default=True)
            If True, displays the visualization in a Matplotlib window.

       Returns
        -------
        None
        """

        axis.clear()

        G = nx.DiGraph()

        # Add nodes
        for node_id, node_type in network.nodes.items():
            G.add_node(node_id, label=node_type)

        # Add edges
        for (src, tgt), conn in network.connections.items():
            if conn["enabled"]:
                G.add_edge(src, tgt, weight=conn["weight"])

        # Layered layout (input -> hidden -> output)
        layers = {"input": [], "hidden": [], "output": []}
        for node_id, node_type in network.nodes.items():
            layers[node_type].append(node_id)

        pos = {}
        for i, node_id in enumerate(sorted(layers["input"])):
            pos[node_id] = (0, -i)
        for i, node_id in enumerate(sorted(layers["hidden"])):
            pos[node_id] = (1, -i)
        for i, node_id in enumerate(sorted(layers["output"])):
            pos[node_id] = (2, -i)

        # Colors
        node_colors = []
        for node_id in G.nodes:
            t = network.nodes[node_id]
            if t == "input":
                node_colors.append("skyblue")
            elif t == "output":
                node_colors.append("lightgreen")
            else:
                node_colors.append("orange")

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=axis)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=14, ax=axis)
        nx.draw_networkx_labels(
            G, pos,
            {n: f"{n}\n{network.nodes[n]}" for n in G.nodes},
            font_size=8,
            ax=axis
        )

        # Edge labels (weights)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=axis)

        # Saving
        file = Path(self.save_folder, "individuals", f"{filename}.png")
        if save_image:
            file.parent.mkdir(parents=True, exist_ok=True)
            axis.figure.savefig(file, bbox_inches="tight", dpi=150)

        # Display
        if show_image:
            plt.show()

