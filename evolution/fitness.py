from __future__ import annotations
from typing import TYPE_CHECKING
import abc

from core.orchestrate import run_population
from core.types import GeneticAlgorithmParams
from evolution.neural_network import NeuralNetwork

if TYPE_CHECKING:
    from core.types import RunResult


def evaluate_population(population: list[NeuralNetwork], params: GeneticAlgorithmParams) -> list[NeuralNetwork]:
    """Calculates fitness function values for a population of neural networks.

    Args:
        population (list[NeuralNetwork]): The population to calculate the fintess values of.
        params (GeneticAlgorithmParams): GA parameters, including the fitness function to use.

    Returns:
        npt.NDArray[np.float32]: A list of fitness values for respective individuals.
    """
    run_results = run_population(population, params)

    fitness_function = params.fitness

    for indiv, run_result in zip(population, run_results):
        indiv.fitness_value = fitness_function.calculate(run_result)

    return population


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