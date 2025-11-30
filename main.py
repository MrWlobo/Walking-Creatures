from pathlib import Path
from core.evolve import GeneticAlgorithm
from core.types import FullJointStateGetter, GeneticAlgorithmParams, TimeOnlyRunConditions
from evolution.fitness import XDistanceFitness
from evolution.selection import TournamentSelection


def run_genetic_algorithm(params: GeneticAlgorithmParams):
    ga = GeneticAlgorithm(params)
    
    ga.evolve()


if __name__ == "__main__":
    params = GeneticAlgorithmParams(
        creature_path=Path(__file__).parent / "assets/creatures" / "1-spherical_hopper.urdf",
        results_path=Path(__file__).parent / "results",
        
        fitness=XDistanceFitness(),
        selection=TournamentSelection(tournament_size=5),
        state_getter=FullJointStateGetter(),
        run_conditions=TimeOnlyRunConditions(max_time_seconds=10),
        
        population_size=1_000,
        n_generations=10,
        
        genetic_operation_ratios=(0.9, 0.05, 0.05),
        mutation_type_percentages=[75, 20, 5],
        
        indiv_output_scale=5,
        
        speciation_coefficients=(1.2, 1.4, 0.6),
        speciation_compatibility_distance=1.5,
        
        n_processes=None
    )
    
    run_genetic_algorithm(params)