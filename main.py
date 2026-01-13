from pathlib import Path
from scipy.special import softmax

from core.evolve import GeneticAlgorithm
from core.types import BaseAndFullJointStateGetter, FallOrTimeoutRunConditions, FullJointStateGetter, GeneticAlgorithmParams, TimeOnlyRunConditions
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
        state_getter=BaseAndFullJointStateGetter(),
        run_conditions=FallOrTimeoutRunConditions(max_time_seconds=10, height_threshold=0.4),
        
        population_size=10,
        n_generations=500,
        
        initial_connections=40,
        
        succession_ratio=0.0,
        genetic_operation_ratios=lambda i: (0.0, 1.0) if i < 20 else (softmax([0.01*i, 1]) if i < 184 else (0.7, 0.3)),
        mutation_type_percentages=[60, 37.5, 2.5],
        weight_mutation_params=(0.6, -2.2, 2.2, -10.0, 10.0),
        mutation_after_crossover_probability=0.15,
        
        indiv_output_scale=7.0,
        
        speciation_coefficients=(1.5, 1.2, 0.04),
        speciation_compatibility_distance=1.4,
        
        n_processes=None
    )
    
    run_genetic_algorithm(params)