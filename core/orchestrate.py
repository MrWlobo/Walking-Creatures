import numpy as np

from evolution.neural_network import NeuralNetwork
from simulation.simulation import Simulation
from core.data_utils import CreatureStateGetter
from core.types import RunResult, RunConditions


def run_individual(indiv: NeuralNetwork, sim: Simulation, state_getter: CreatureStateGetter, indiv_output_scale: float, run_condidtions: RunConditions):
    sim.reset_state()

    n_revolute = sim.num_revolute
    n_spherical = sim.num_spherical

    revolute_indices = sim.revolute_joints
    spherical_indices = sim.spherical_joints


    while not run_condidtions.isRunEnd(sim):
        creature_state = state_getter.get_state(sim)
        
        indiv_output = indiv.forward(creature_state)

        # temporary
        indiv_output = np.array(indiv_output)
        #
        
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


