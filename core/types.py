from __future__ import annotations
from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import abc
from pathlib import Path

from simulation.simulation import Simulation
from evolution.fitness import Fitness
from evolution.selection import Selection

class GeneticAlgorithmParams:
    """Customizable parameters for the genetic algorithm. They should be consistent across the whole GA, as they are passed between many functions.

    Attributes:
        creature_path (Path): Absolute path to .urdf creature model to use in the algorithm.
        fitness (Fitness): The fitness function to base the GA on.
        selection (Selection): The method of selecting best individuals from a population to use.     
        state_getter (CreatureStateGetter): An object defining what data to extract from the simulation every tick 
                                            and to feed to the individual's neural network.
        run_conditions (RunConditions): Defines the conditions for the simulation to end (e.g. time limit).
        population_size (int): Number of individuals in each generation.
        indiv_output_scale (float): Value to multiply NN outputs by before passing them to joint control functions.
                                    Used to match the appropriate order of magnitude of forces for the particular creature.
        speciation_coefficients (tuple[float, float, float]): Weights for (excess, disjoint, weight differences)
                                                                in compatibility calculation for speciation purposes.
        speciation_compatibility_distance (float): Threshold for determining whether two networks belong to the same species.
        n_processes (int): How many threads to use when running the population's simulations. 
                            Use None to let the functions use the maximum available number of threads.
        time_step (float): How much time should one tick of the simulation represent. Probably not necessary to change.
        settle_steps (int): Number of steps to wait before letting the simulation settle down before saving the initial state.
                            Most likely not necessary to ever change.
    """
    creature_path: Path
    fitness: Fitness
    selection: Selection
    state_getter: CreatureStateGetter
    run_conditions: RunConditions
    population_size: int
    indiv_output_scale: float
    speciation_coefficients: tuple[float, float, float]
    speciation_compatibility_distance: float
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
        self.max_time_seconds = max_time_seconds
    

    def isRunEnd(self, sim: Simulation) -> bool:
        return sim.tick_count * sim.time_step >= self.max_time_seconds


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


def _combine_inputs(arrays) -> npt.NDArray:
    """
    Takes N inputs (arrays or lists), flattens them, 
    and combines them into a single 1D NumPy array.
    """
    processed_list = [np.asanyarray(x).ravel() for x in arrays]
    
    return np.concatenate(processed_list)
