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
        creature_path=Path("assets/creatures") / "6-revolute_biped.urdf",
        surface_friction=0.7,
        
        results_path=Path("results"),
        
        fitness=XDistanceStabilityFitness(stability_coefficient=0.001),
        selection=TournamentSelection(tournament_size=10),
        state_getter=FullJointStateGetter(),
        run_conditions=FallOrTimeoutRunConditions(max_time_seconds=10, height_threshold=0.2),
        
        population_size=2_000,
        n_generations=500,
        
        initial_connections=4,
        
        succession_ratio=0.005,
        genetic_operation_ratios=(0.7, 0.3),
        mutation_type_percentages=[60, 37.5, 2.5],
        weight_mutation_params=(0.01, -2.2, 2.2, -10.0, 10.0),
        
        indiv_output_scale=7.0,
        
        speciation_coefficients=(2.0, 1.7, 0.4),
        speciation_compatibility_distance=1.8,
        
        n_processes=None
    )
    
    run_genetic_algorithm(params)