import math
import random
from typing import Callable
from collections import defaultdict, deque
import numpy as np


class NeuralNetwork:
    """
    A NEAT-style neural network implementation with support for:
    - Arbitrary hidden nodes
    - Connection and node mutations with global innovation tracking
    - Forward propagation using topological sorting
    - 1D and 3D output units
    """

    connection_split_table = {} # connection: id of node that splits the connection


    def __init__(self, input_units: int = None, units_1d: int = None, units_3d: int = None, activation_function: Callable = math.tanh, nodes: dict[int, str] = None, connections: dict[tuple[int, int], dict[str, float | bool]] = None):
        """
        Initialize the neural network with input and output units.

        Args:
            input_units (int): Number of input neurons.
            units_1d (int): Number of 1D output units.
            units_3d (int): Number of 3D output units (each counts as 3 neurons).
            activation_function (Callable): Activation function applied to hidden/output nodes.
        """

        if nodes is not None and connections is not None:
            self.input_units = len([n for n, t in nodes.items() if t == "input"])
            self.output_units = len([n for n, t in nodes.items() if t == "output"])
            self.nodes = nodes
            self.connections = connections
            self.activation_function = activation_function
            self.fitness_value = None
        else:
            self.input_units = input_units
            self.output_units = units_1d + 3 * units_3d
            self.nodes = {**{k: "input" for k in range(self.input_units)}, **{k: "output" for k in range(self.input_units, self.output_units + self.input_units)}}
            self.connections = {}
            self.activation_function = activation_function
            self.fitness_value = None

            for input in range(self.input_units):
                for output in range(self.input_units, self.output_units + self.input_units):
                    self.connections[(input, output)] = {"weight": random.uniform(-1,1), "enabled": random.choice([True, False])}


    def change_weight(self, connection: tuple[int, int], new_weight: float) -> None:
        """
        Update the weight of an existing connection.

        Args:
            connection (tuple[int, int]): Source and target node indices.
            new_weight (float): New weight value.
        """

        self.connections[connection]["weight"] = new_weight


    def toggle_connection(self, connection) -> None:
        """
        Toggle the enabled/disabled state of a connection.

        Args:
            connection (tuple[int, int]): A tuple representing the source and target node indices
                                            of the connection to toggle.

        Modifies:
            self.connections[connection]["enabled"]: Flips True to False or False to True.
        """

        self.connections[connection]["enabled"] = not self.connections[connection]["enabled"]


    def add_connection(self, connection) -> bool:
        """
        Add a new connection between nodes if allowed.

        Args:
            connection (tuple[int, int]): Source and target node indices.

        Returns:
            bool: True if connection was successfully added, False otherwise.
        """

        if connection in self.connections:
            return False

        if connection[0] not in self.nodes or connection[1] not in self.nodes:
            return False

        if self.nodes[connection[0]] == "output" or self.nodes[connection[1]] == "input":
            return False

        self.connections[connection] = {"weight": random.uniform(-1, 1), "enabled": True}

        # Check for cycle
        topo_order = self._topological_sort()
        if len(topo_order) != len(self.nodes):
            # Cycle detected, remove the temporary connection
            del self.connections[connection]
            return False

        return True


    def add_node_to_connection(self, connection) -> bool:
        """
        Split an existing connection by adding a hidden node.

        Args:
            connection (tuple[int, int]): Connection to split.

        Returns:
            bool: True if a node was successfully added, False otherwise.
        """

        if connection not in self.connections:
            return False

        # Add a new node to the network, check its index in the global table, create it if it does not exist.
        if connection not in NeuralNetwork.connection_split_table:
            NeuralNetwork.connection_split_table[connection] = max(NeuralNetwork.connection_split_table.values(), default=len(self.nodes)-1) + 1
        node = NeuralNetwork.connection_split_table[connection]
        self.nodes[node] = "hidden"

        # Disable the original connection
        self.connections[connection]["enabled"] = False

        # Create new connections such that they do not change networks previous behaviour (hence 1.0 weight).
        self.connections[(connection[0], node)] = {"weight": self.connections[connection]["weight"], "enabled": True}
        self.connections[(node, connection[1])] = {"weight": 1.0, "enabled": True}

        return True


    def forward(self, input_values: np.array) -> np.array:
        """
        Perform a forward pass through the network.

        Args:
            input_values (np.array): Input vector of length equal to number of input units.

        Returns:
            np.array: Output vector after forward propagation.
        """
        node_values = {n: 0.0 for n in self.nodes}

        # Assign input values
        input_nodes = [n for n, t in self.nodes.items() if t == "input"]
        for n, val in zip(input_nodes, input_values):
            node_values[n] = val

        # Topological sort based on enabled connections
        order = self._topological_sort()

        # Compute node activations
        for node in order:
            if self.nodes[node] == "input":
                continue

            incoming = [
                (src, conn["weight"])
                for (src, tgt), conn in self.connections.items()
                if tgt == node and conn["enabled"]
            ]

            if incoming:
                sources, weights = zip(*incoming)
                src_vals = np.fromiter((node_values[s] for s in sources), float)
                w = np.fromiter(weights, float)

                total = np.dot(src_vals, w)
            else:
                total = 0.0

            node_values[node] = self.activation_function(total)

        # Collect outputs
        output_nodes = [n for n, t in self.nodes.items() if t == "output"]
        return np.array([node_values[n] for n in output_nodes])


    def _topological_sort(self):
        """
        Perform a topological sort of the network nodes based on enabled connections.

        Returns:
            list[int]: Nodes in topologically sorted order.
        """
        indegree = {n: 0 for n in self.nodes}
        adjacency = defaultdict(list)
        for (src, tgt), conn in self.connections.items():
            if conn["enabled"]:
                adjacency[src].append(tgt)
                indegree[tgt] += 1

        queue = deque([n for n in self.nodes if indegree[n] == 0])
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adjacency[n]:
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)
        return order


    def __repr__(self):
        result = ""
        for key in self.connections:
            result += f"{key}: {self.connections[key]}\n"

        return result
