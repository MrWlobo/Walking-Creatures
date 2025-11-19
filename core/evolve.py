from pathlib import Path

from evolution.fitness import Fitness
from evolution.selection import Selection
from core.data_utils import CreatureStateGetter
from core.types import RunConditions

class GeneticAlgorithmParams:
    creature_path: Path
    fitness: Fitness
    selection: Selection
    state_getter: CreatureStateGetter
    run_conditions: RunConditions


class GeneticAlgorithm:
    def __init__(self, params: GeneticAlgorithmParams):
        pass