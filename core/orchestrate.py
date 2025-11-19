import multiprocessing
import pybullet as p
from pathlib import Path
from functools import partial

from evolution.neural_network import NeuralNetwork
from simulation.simulation import Simulation
from core.data_utils import CreatureStateGetter
from core.types import RunResult, RunConditions


_worker_sim = None

def run_population(population: list[NeuralNetwork], creature_path: Path, state_getter: CreatureStateGetter, indiv_output_scale: float, run_condidtions: RunConditions, n_processes: int = None):
    with multiprocessing.Pool(
        processes=n_processes,
        initializer=_init_process,
        initargs=(p.DIRECT, creature_path.resolve(), 120, 1./240.,)
    ) as pool:
        
        _run = partial(
            _run_process,
            state_getter=state_getter,
            indiv_output_scale=indiv_output_scale,
            run_condidtions=run_condidtions
        )

        run_results = pool.map(_run, population)
    
    return run_results


def _init_process(sim_type, creature_path, settle_steps, time_step):
    """Initializes the simulation inside the worker process."""
    global _worker_sim

    _worker_sim = Simulation(
        simulation_type=sim_type,
        creature_path=creature_path,
        settle_steps=settle_steps,
        time_step=time_step
    )


def _run_process(indiv: NeuralNetwork, state_getter: CreatureStateGetter, indiv_output_scale: float, run_condidtions: RunConditions):
    return run_individual(indiv, _worker_sim, state_getter, indiv_output_scale, run_condidtions)


def run_individual(indiv: NeuralNetwork, sim: Simulation, state_getter: CreatureStateGetter, indiv_output_scale: float, run_condidtions: RunConditions):
    sim.reset_state()

    n_revolute = sim.num_revolute
    n_spherical = sim.num_spherical

    revolute_indices = sim.revolute_joints
    spherical_indices = sim.spherical_joints


    while not run_condidtions.isRunEnd(sim):
        creature_state = state_getter.get_state(sim)
        
        indiv_output = indiv.forward(creature_state)

        indiv_output = indiv_output * indiv_output_scale

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
