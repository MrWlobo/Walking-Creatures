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
    creature_path: Path
    fitness: Fitness
    selection: Selection
    state_getter: CreatureStateGetter
    run_conditions: RunConditions
    indiv_output_scale: float
    n_processes: int = None
    time_step: float = 1./240.
    settle_steps: int = 120


@dataclass
class RunResult:
    """Holds relevant results from a single simulation run."""
    time_seconds: float
    final_position: npt.NDArray[np.float64]


# run conditions
#################

class RunConditions(abc.ABC):
    @abc.abstractmethod
    def isRunEnd(self, sim: Simulation) -> bool:
        pass


class TimeOnlyRunConditions(RunConditions):
    def __init__(self, max_time_seconds: float):
        self.max_time_seconds = max_time_seconds
    

    def isRunEnd(self, sim: Simulation) -> bool:
        return sim.tick_count * sim.time_step >= self.max_time_seconds


# creature state getters
#################

class CreatureStateGetter(abc.ABC):
    @abc.abstractmethod
    def get_state(self, simulation: Simulation) -> list[np.ndarray]:
        pass


class FullJointStateGetter(CreatureStateGetter):
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
