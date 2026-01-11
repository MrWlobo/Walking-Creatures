import random
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axis import Axis

from evolution.neural_network import NeuralNetwork


class Visualization:

    species_colors = {}
    species_markers = {}
    marker_shapes = ["o", "s", "^", "D", "v", "<", ">", "*", "P", "X"]

    def __init__(self, save_folder: str):
        self.save_folder = save_folder

    def visualize_network(self, network: NeuralNetwork, axis: Axis, filename: str, title: str = None,
                          save_image: bool = False, show_image: bool = True) -> None:
        """
        Visualize a NEAT-style neural network with edge thickness proportional to |weight|
        and color indicating sign (red=negative, green=positive).
        """

        axis.clear()

        if title is None:
            axis.set_title(f"Individual with a fitness value of {network.fitness_info}")
        else:
            axis.set_title(title)

        G = nx.DiGraph()

        # Add nodes
        for node_id, node_type in network.nodes.items():
            G.add_node(node_id, label=node_type)

        # Add edges (only enabled)
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

        # Node colors
        node_colors = []
        for node_id in G.nodes:
            t = network.nodes[node_id]
            if t == "input":
                node_colors.append("skyblue")
            elif t == "output":
                node_colors.append("lightgreen")
            else:
                node_colors.append("orange")

        # Edge colors and widths
        edges = G.edges(data=True)
        edge_weights = [abs(d['weight']) for u, v, d in edges]

        max_width = 2.4
        min_width = 0.8
        widths = []
        edge_colors = []

        max_weight = max(edge_weights) if edge_weights else 1
        for u, v, d in edges:
            w_abs = abs(d['weight'])
            width = min_width + (w_abs / max_weight) * (max_width - min_width)
            widths.append(width)

            if d['weight'] >= 0:
                intensity = 0.3 + 0.7 * min(1.0, d['weight'] / max_weight)
                edge_colors.append((0.2, intensity, 0.2))
            else:
                intensity = 0.3 + 0.7 * min(1.0, abs(d['weight']) / max_weight)
                edge_colors.append((intensity, 0.2, 0.2))

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=axis)
        nx.draw_networkx_edges(G, pos, arrowstyle="-", arrowsize=14, width=widths, edge_color=edge_colors, ax=axis)
        nx.draw_networkx_labels(
            G, pos,
            {n: f"{n}\n{network.nodes[n]}" for n in G.nodes},
            font_size=8,
            ax=axis
        )

        # Save
        file = Path(self.save_folder, f"{filename}.png")
        if save_image:
            file.parent.mkdir(parents=True, exist_ok=True)
            axis.figure.savefig(file, bbox_inches="tight", dpi=150)

        # Display
        if show_image:
            plt.show()

    def visualize_population(self, population: list[list[NeuralNetwork]], axis: Axis, filename: str, title: str = None, save_image: bool = False, show_image: bool = True) -> None:
        """
        Visualize a population of NEAT networks as a scatter plot, coloring and shaping points by species.

        Each species is assigned a unique color and marker shape. Individuals are plotted at
        random positions within a fixed range for visual separation.

        X and Y axis ticks are removed for clarity. The plot can optionally be saved or displayed.

        Parameters
        ----------
        population : list of list of NeuralNetwork
            The population of neural networks, divided into species.
            Each sublist represents a species containing its individuals.

        axis : matplotlib.axes.Axes
            The axes on which to draw the scatter plot.

        filename : str
            Base filename used when saving the image. The file will be saved under
            `self.save_folder/populations/<filename>.png`.

        title : str, optional (default=None)
            Title for the plot. If None, a default title showing the number of individuals
            and species will be used.

        save_image : bool, optional (default=False)
            If True, the plot will be saved to the `self.save_folder`.

        show_image : bool, optional (default=True)
            If True, the plot will be displayed using Matplotlib.

        Returns
        -------
        None
            The function only draws the plot, optionally saves or shows it, and does not return anything.
        """

        min_pos, max_pos = 1, 1000
        color_value_reduce = 0.2

        axis.set_xticks([])
        axis.set_yticks([])

        if title is None:
            axis.set_title(f"Population of {sum([len(species) for species in population])} individuals divided into {len(population)} species")
        else:
            axis.set_title(title)

        for i, species in enumerate(population):

            if i in Visualization.species_colors:
                color = Visualization.species_colors[i]
                marker = Visualization.species_markers[i]
            else:
                color = (abs(random.random() - color_value_reduce), abs(random.random() - color_value_reduce), abs(random.random() - color_value_reduce))
                marker = random.choice(Visualization.marker_shapes)
                Visualization.species_colors[i] = color
                Visualization.species_markers[i] = marker

            for individual in species:
                axis.scatter(random.uniform(min_pos, max_pos), random.uniform(min_pos, max_pos), color=color, marker=marker)

        file = Path(self.save_folder, f"{filename}.png")
        if save_image:
            file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file, bbox_inches="tight", dpi=150)

        if show_image:
            plt.show()


    def visualize_population_with_fittest_individual(self, population: list[list[NeuralNetwork]], current_generation: int, filename: str, save_image: bool = False, show_image: bool = True) -> None:
        """
        Visualize a population of NEAT networks alongside the fittest individual.

        The function creates a side-by-side plot with two subplots:
        1. A scatter plot of the population, with different colors and markers for species.
        2. A detailed network visualization of the fittest individual in the population.

        Parameters
        ----------
        population : list of list of NeuralNetwork
            The population divided into species. Each sublist contains individuals of one species.

        current_generation : int
            The current generation number, used for the plot title.

        filename : str
            Base filename for saving the figure. The plot will be saved under
            `self.save_folder/populations_with_best_individual/<filename>.png`.

        save_image : bool, optional (default=False)
            If True, saves the plot to the specified folder.

        show_image : bool, optional (default=True)
            If True, displays the plot with Matplotlib.

        Raises
        ------
        ValueError
            If the population contains no individuals.

        Returns
        -------
        None
            The function produces a visualization and optionally saves or displays it.
        """

        fittest = None
        for species in population:
            for individual in species:
                if fittest is None:
                    fittest = individual
                elif fittest.fitness_info < individual.fitness_info:
                    fittest = individual

        if fittest is None:
            raise ValueError("There are no individuals in the given population.")

        figure, axis = plt.subplots(1, 2, figsize=(12, 6))
        self.visualize_population(population, axis[0], "", f"Population of {sum([len(species) for species in population])} individuals divided into {len(population)} species", False, False)
        self.visualize_network(fittest, axis[1], "", f"Best individual with fitness {fittest.fitness_info}", False, False)
        plt.suptitle(f"Generation {current_generation}", fontweight='bold', fontsize=14)

        file = Path(self.save_folder, f"{filename}.png")
        if save_image:
            file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file, bbox_inches="tight", dpi=150)

        if show_image:
            plt.show()
        
        plt.close(figure)

    def line_plot(self, values: list[float], axis: Axis, title: str, y_label: str, filename: str, save_image: bool = False, show_image: bool = True) -> None:
        """
        Plot a line graph of a metric across generations.

        The function creates a simple line plot showing how a given metric evolves
        over time.

        Parameters
        ----------
        values : list of float
            A sequence of numerical values to plot, one for each generation.

        axis : matplotlib.axes.Axes
            The axes on which to draw the scatter plot.

        title : str
            The title displayed above the plot.

        y_label : str
            Label for the y-axis, describing the metric being plotted.

        filename : str
            Base filename for saving the image. The plot will be saved under
            `self.save_folder/<filename>.png` if `save_image` is True.

        save_image : bool, optional (default=False)
            If True, saves the plot to disk.

        show_image : bool, optional (default=True)
            If True, displays the plot using Matplotlib.

        Raises
        ------
        ValueError
            If `values` is empty.

        Returns
        -------
        None
            The function generates a plot and optionally saves or displays it.
        """

        if not values:
            raise ValueError("There are no values.")

        X = list(range(0, len(values)))
        Y = values

        axis.plot(X, Y)
        axis.set_xlabel("Generation")
        axis.set_ylabel(y_label)
        axis.set_title(title)

        file = Path(self.save_folder, f"{filename}.png")
        if save_image:
            file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(file, bbox_inches="tight", dpi=150)

        if show_image:
            plt.show()

