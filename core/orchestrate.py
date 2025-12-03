import multiprocessing
import pybullet as p
from functools import partial
import atexit

from evolution.neural_network import NeuralNetwork
from simulation.simulation import Simulation
from core.types import GeneticAlgorithmParams, RunResult


_worker_sim = None

def run_population(population: list[NeuralNetwork], params: GeneticAlgorithmParams) -> list[RunResult]:
    """Runs creature walking simulations based on all individuals in the population and returns relevant run data.

    Args:
        population (list[NeuralNetwork]): The population of neural networks to use to run creature simulations.
        params (GeneticAlgorithmParams): See GeneticAlgorithmParams documentation. 
                                            The params should be consistent across the whole GA.

    Returns:
        list[RunResult]: Run data for all individuals in the population, ordered the same way as the population list.
    """
    with multiprocessing.Pool(
        processes=params.n_processes,
        initializer=_init_process,
        initargs=(p.DIRECT, params)
    ) as pool:
        
        _run = partial(
            _run_process,
            params=params
        )

        run_results = pool.map(_run, population)
    
    return run_results


def run_individual(indiv: NeuralNetwork, sim: Simulation, params: GeneticAlgorithmParams) -> RunResult:
    """Runs a creature walking simulation based on an individual and returns relevant run data.
    The function is designed to work on a perviously initialized Simulation object, for efficiency.

    Args:
        indiv (NeuralNetwork): The individual to base the simulation on.
        sim (Simulation): The Simulation object to use for the simulation. Due to the huge overhead of initializing the physics engine,
                            they should be reused as much as possble.
        params (GeneticAlgorithmParams): The parameters of the genetic algorithm. See GeneticAlgorithmParams documentation.

    Returns:
        RunResult: Run data.
    """
    # ensure that the same individual always get the same random seed for jitter, for reproducibility
    sim_seed = hash(indiv.__repr__()) % (2**32)
    sim.reset_state(seed=sim_seed)

    n_revolute = sim.num_revolute
    n_spherical = sim.num_spherical

    revolute_indices = sim.revolute_joints
    spherical_indices = sim.spherical_joints


    while not params.run_conditions.isRunEnd(sim):
        creature_state = params.state_getter.get_state(sim)
        
        indiv_output = indiv.forward(creature_state)

        indiv_output = indiv_output * params.indiv_output_scale

        revolute_target = indiv_output[:n_revolute]
        spherical_target = indiv_output[n_revolute:].reshape(n_spherical, 3)

        sim.moveRevolute(revolute_indices, revolute_target)
        sim.moveSpherical(spherical_indices, spherical_target)

        sim.step()
    
    final_time = sim.tick_count * sim.time_step
    final_position = sim.get_base_state()[0]

    return RunResult(
        time_seconds=final_time,
        final_position=final_position
    )


def _init_process(sim_type, params: GeneticAlgorithmParams):
    """Initializes the simulation inside the worker process."""
    global _worker_sim

    atexit.register(_cleanup_process)

    _worker_sim = Simulation(
        simulation_type=sim_type,
        creature_path=params.creature_path,
        settle_steps=params.settle_steps,
        time_step=params.time_step
    )


def _cleanup_process():
    global _worker_sim
    
    _worker_sim.terminate()


def _run_process(indiv: NeuralNetwork, params: GeneticAlgorithmParams):
    return run_individual(indiv, _worker_sim, params)
