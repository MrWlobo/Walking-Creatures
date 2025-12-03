from pathlib import Path
from core.evolve import GeneticAlgorithm
from core.types import FallOrTimeoutRunConditions, FullJointStateGetter, GeneticAlgorithmParams, TimeOnlyRunConditions
from evolution.fitness import XDistanceFitness, XDistanceStabilityFitness
from evolution.selection import TournamentSelection


def run_genetic_algorithm(params: GeneticAlgorithmParams):
    ga = GeneticAlgorithm(params)
    
    ga.evolve()


if __name__ == "__main__":
    params = GeneticAlgorithmParams(
        creature_path=Path(__file__).parent / "assets/creatures" / "6-revolute_biped.urdf",
        results_path=Path(__file__).parent / "results",
        
        fitness=XDistanceStabilityFitness(stability_coefficient=0.01),
        selection=TournamentSelection(tournament_size=10),
        state_getter=FullJointStateGetter(),
        run_conditions=FallOrTimeoutRunConditions(max_time_seconds=10, height_threshold=0.3),
        
        population_size=1_000,
        n_generations=300,
        
        initial_connections=2,
        
        succession_ratio=0.005,
        genetic_operation_ratios=(0.7, 0.3),
        mutation_type_percentages=[60, 30, 10],
        
        indiv_output_scale=0.05,
        
        speciation_coefficients=(4.6, 1.2, 0.4),
        speciation_compatibility_distance=1.5,
        
        n_processes=None
    )
    
    run_genetic_algorithm(params)