from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import abc
import numpy as np

from core.orchestrate import run_population
from evolution.exceptions import NoneFitnessException
from evolution.neural_network import NeuralNetwork

if TYPE_CHECKING:
    from core.types import RunResult
    from core.types import GeneticAlgorithmParams


class Fitness(abc.ABC):
    """
    Abstract Base Class for all fitness functions.
    """
    
    @abc.abstractmethod
    def calculate(self, run_result: RunResult) -> float:
        """
        Calculates the fitness from a RunResult object.
        
        Important: fitness functions should be designed in such a way,
        that higher values are assigned to better individuals and
        lower values are assigned to worse individuals.
        
        :param RunResult run_result: A dataclass containing run data.
        :return float: A single float value representing the fitness.
        """
        pass


    @abc.abstractmethod
    def adjustSpeciesFitness(self, species: list[list[NeuralNetwork]]) -> list[list[NeuralNetwork]]:
        """
        Adjust the fitness of individuals based on their species' size, making it competition
        between species of different sizes balanced.
        
        :param species: list[list[NeuralNetwork]] species: The speciated population (with set fitness values).
        :return list[list[NeuralNetwork]]: The speciated population with adjusted fitness values.
        """
        pass


    @abc.abstractmethod
    def getStats(self, population: list[NeuralNetwork]) -> FitnessStats:
        """
        Calculates fitness stats for a population (or species).
        
        :param list[NeuralNetwork] population: The population or species (with set fitness values).
        :return FitnessStats: An object containing the fitness stats.
        """
        pass


class XDistanceFitness(Fitness):
    """
    A fitness function that rewards distance traveled
    along the positive X-axis.
    """
    def calculate(self, run_result: RunResult) -> float:
        return run_result.final_position[0]
    

    def adjustSpeciesFitness(self, species: list[list[NeuralNetwork]]) -> list[list[NeuralNetwork]]:
        for s in species:
            curr_size = len(s)

            for indiv in s:
                indiv.fitness_value /= curr_size
        
        return species
    

    def getStats(self, population: list[NeuralNetwork]) -> FitnessStats:
        if any([i is None for i in population]):
            raise NoneFitnessException(f"Encountered None fitness value, population: {population}")
        
        fitness = np.array([indiv.fitness_value for indiv in population])

        return FitnessStats(
            best_fitness=np.max(fitness),
            mean_fitness=np.mean(fitness),
        )



@dataclass
class FitnessStats:
    best_fitness: float
    mean_fitness: float


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