from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import abc
from pathlib import Path

if TYPE_CHECKING:
    import numpy.typing as npt
    from simulation.simulation import Simulation
    from evolution.fitness import Fitness
    from evolution.selection import Selection

@dataclass
class GeneticAlgorithmParams:
    """Customizable parameters for the genetic algorithm. They should be consistent across the whole GA, as they are passed between many functions.

    Attributes:
        creature_path (Path): Path (relative to the project path) to .urdf creature model to use in the algorithm.
        results_path (Path): Path (relative to the project path) to a directory in which a new folder containing the GA run's results will be created.
        fitness (Fitness): The fitness function to base the GA on.
        selection (Selection): The method of selecting best individuals from a population to use.     
        state_getter (CreatureStateGetter): An object defining what data to extract from the simulation every tick 
                                            and to feed to the individual's neural network.
        run_conditions (RunConditions): Defines the conditions for the simulation to end (e.g. time limit).
        population_size (int): Number of individuals in each generation.
        n_generations (int): Number of generations to rune the genetic algorithm for.
        initial_connections (int): Number of initial random connections between neurons of each individual.
        succession_ratio (float | Callable[[int], float]): Percentage of top individuals to copy to the next population each generation.
                                                            If a function is passed, it should return a float for each generation number it gets as an argument.
        genetic_operation_ratios (tuple[float, float] | Callable[[int], tuple[float, float]]): Probabilities of applying crossover and mutation
                                                                operations, respectively.
                                                                If a function is passed, it should return a tuple of 2 floats for each generation number it gets as an argument.
        mutation_type_percentages (list[float] | Callable[[int], list[float]]): List of 3 integers representing percent chance for 
                                                    [0] mutate weight, [1] mutate connection, [2] mutate node. Must sum to 100.
                                                    If a function is passed, it should return a list of 3 integers for each generation number it gets as an argument.
        weight_mutation_params (tuple[float, float, float, float, float]): Parameters controlling weight mutation
                                                                        * **0**: Probability of perturbing vs replacing a weight.
                                                                        * **1**: Minimum additive perturbation.
                                                                        * **2**: Maximum additive perturbation.
                                                                        * **3**: Minimum value for full weight replacement.
                                                                        * **4**: Maximum value for full weight replacement.
        mutation_after_crossover_probability (float | Callable[[int], float]): Probability of performing and additional mutation for offspring genearted via crossover.
                                                                                If a function is passed, it should return a float for each generation number it gets as an argument.
        indiv_output_scale (float): Value to multiply NN outputs by before passing them to joint control functions.
                                    Used to match the appropriate order of magnitude of forces for the particular creature.
        speciation_coefficients (tuple[float, float, float]): Weights for (excess, disjoint, weight differences)
                                                                in compatibility calculation for speciation purposes.
        speciation_compatibility_distance (float): Threshold for determining whether two networks belong to the same species.
        surface_friction (float): Friction of the surface the creatures walk on.
        n_processes (int): How many threads to use when running the population's simulations. 
                            Use None to let the functions use the maximum available number of threads.
        time_step (float): How much time should one tick of the simulation represent. Probably not necessary to change.
        settle_steps (int): Number of steps to wait before letting the simulation settle down before saving the initial state.
                            Most likely not necessary to ever change.
    """
    creature_path: Path
    results_path: Path
    fitness: Fitness
    selection: Selection
    state_getter: CreatureStateGetter
    run_conditions: RunConditions
    population_size: int
    n_generations: int
    initial_connections: int
    succession_ratio: float | Callable[[int], float]
    genetic_operation_ratios: tuple[float, float] | Callable[[int], tuple[float, float]]
    mutation_type_percentages: list[float] | Callable[[int], list[float]]
    weight_mutation_params: tuple[float, float, float, float, float]
    mutation_after_crossover_probability: float | Callable[[int], float]
    indiv_output_scale: float
    speciation_coefficients: tuple[float, float, float]
    speciation_compatibility_distance: float
    surface_friction: float = 0.7
    n_processes: int = None
    time_step: float = 1./240.
    settle_steps: int = 120


@dataclass
class RunResult:
    """Holds relevant results from a single simulation run.

    Attributes:
        time_seconds (float): How much time (in seconds) it took for the simulation to finish.
        final_position (npt.NDArray[np.float64]): The final position ([x, y, z]) of the creature.
    """
    time_seconds: float
    final_position: npt.NDArray[np.float64]


# run conditions
#################

class RunConditions(abc.ABC):
    """Abstract class for run conditions definitons.
        These classes are meant to define conditions which should be true for the simulation to end.
        Example: a time limit, so the simulation doesn't run indefinitely.
    """
    @abc.abstractmethod
    def isRunEnd(self, sim: Simulation) -> bool:
        pass


class TimeOnlyRunConditions(RunConditions):
    """A RunConditions impelementation that sets a time limit for the simulations.
    """
    def __init__(self, max_time_seconds: float):
        """Initializes a TimeOnlyRunConditions object.

        Args:
            max_time_seconds (float): _description_
        """
        self.max_time_seconds = max_time_seconds
    

    def isRunEnd(self, sim: Simulation) -> bool:
        return sim.tick_count * sim.time_step >= self.max_time_seconds
    
    
    def __repr__(self):
        return f"{type(self).__name__}(max_time_seconds={self.max_time_seconds})"


class FallOrTimeoutRunConditions(RunConditions):
    """A RunConditions impelementation that sets a time limit for the simulations and ends the simulation
    when the creature falls over.
    """
    def __init__(self, max_time_seconds: float, height_threshold: float):
        self.max_time_seconds = max_time_seconds
        self.height_threshold = height_threshold
    

    def isRunEnd(self, sim: Simulation) -> bool:
        is_time_up = sim.tick_count * sim.time_step >= self.max_time_seconds
        has_fallen = sim.get_base_state()[0][2] < self.height_threshold
        
        return is_time_up or has_fallen
    
    
    def __repr__(self):
        return f"{type(self).__name__}(max_time_seconds={self.max_time_seconds}, height_threshold={self.height_threshold})"


# creature state getters
#################

class CreatureStateGetter(abc.ABC):
    """Abstract class for defining what data to extract from the simulation every tick 
        and to feed to the individual's neural network.
    """
    @abc.abstractmethod
    def get_state(self, simulation: Simulation) -> npt.NDArray[np.float64]:
        pass


class FullJointStateGetter(CreatureStateGetter):
    """CreatureStateGetter implementation that extracts all joint information from the simulation
        and returns it as a flat numpy array.
    """
    def get_state(self, simulation: Simulation)  -> npt.NDArray[np.float64]:
        r_positions, r_velocities = simulation.get_revolute_joint_states()
        s_positions, s_velocities = simulation.get_spherical_joint_states()

        return _combine_inputs([r_positions, r_velocities, s_positions, s_velocities])
    
    
    def __repr__(self):
        return f"{type(self).__name__}()"


def _combine_inputs(arrays) -> npt.NDArray:
    """
    Takes N inputs (arrays or lists), flattens them, 
    and combines them into a single 1D NumPy array.
    """
    processed_list = [np.asanyarray(x).ravel() for x in arrays]
    
    return np.concatenate(processed_list)
