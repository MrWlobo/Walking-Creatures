from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import abc

from simulation.simulation import Simulation

@dataclass
class RunResult:
    """Holds relevant results from a single simulation run."""
    time_seconds: float
    final_position: npt.NDArray[np.float64]


class RunConditions(abc.ABC):
    @abc.abstractmethod
    def isRunEnd(self, sim: Simulation) -> bool:
        pass


class TimeOnlyRunConditions(RunConditions):
    def __init__(self, max_time: float):
        self.max_time = max_time
    

    def isRunEnd(self, sim: Simulation) -> bool:
        return sim.tick_count * sim.time_step >= self.max_time