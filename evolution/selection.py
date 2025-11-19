import random
import abc

from evolution.neural_network import NeuralNetwork
from evolution.exceptions import NoneFitnessException


class Selection(abc.ABC):
    """
    Abstract Base Class for all selection functions.
    """
    
    @abc.abstractmethod
    def select(self, population: list[NeuralNetwork]) -> NeuralNetwork:
        pass
        

class TournamentSelection(Selection):
    def __init__(self, tournament_size: int):
        """
        Initialize a TournamentSelection object.

        Args:
            tournament_size (int): Size of the tournament from which to select the best individual.
        """
        self.tournament_size = tournament_size


    def select(self, population: list[NeuralNetwork]) -> NeuralNetwork:
        """
        Select a neural network from a population using tournament selection.

        Args:
            population (list[NeuralNetwork]): The population of neural networks to choose from.

        Raises:
            ValueError: If tournament_size is larger than the population.
            NoneFitnessException: If any of the individuals selected for the tournament have None fitness.

        Returns:
            NeuralNetwork: The selected network based on fitness.
        """
        if self.tournament_size > len(population):
            raise ValueError("Tournament can't be larger than the entire population.")

        tournament = random.sample(population, self.tournament_size)
        
        if any([i is None for i in tournament]):
            raise NoneFitnessException(f"Encountered None fitness value, tournament: {tournament}")

        selected = max(tournament, key=lambda i: i.fitness_value)

        return selected


class RouletteSelection(Selection):
    def select(self, population: list[NeuralNetwork]) -> NeuralNetwork:
        """
        Select a neural network from a population using roulette selection.

        Args:
            population (list[NeuralNetwork]): The population of neural networks to choose from.

        Raises:
            NoneFitnessException: If any of the individuals in the population have None fitness.

        Returns:
            NeuralNetwork: The selected network based on fitness.
        """
        if any([i is None for i in population]):
            raise NoneFitnessException(f"Encountered None fitness value in population.")


        total_fitness = sum([i.fitness_value for i in population])
        weights = [i.fitness_value / total_fitness for i in population]

        selected = random.choices(population, weights=weights, k=1)[0]

        return selected