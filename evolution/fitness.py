import abc

from core.types import RunResult


class Fitness(abc.ABC):
    """
    Abstract Base Class for all fitness functions.
    """
    
    @abc.abstractmethod
    def calculate(self, run_result: RunResult) -> float:
        """
        Calculates the fitness from a RunResult object.
        
        :param RunResult run_result: A dataclass containing run data.
        :return float: A single float value representing the fitness.
        """
        pass


class XDistanceFitness(Fitness):
    """
    A fitness function that rewards distance traveled
    along the positive X-axis.
    """
    def calculate(self, run_result: RunResult) -> float:
        return run_result.final_position[0]