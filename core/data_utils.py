import numpy as np
import numpy.typing as npt
import abc

from simulation.simulation import Simulation


class CreatureStateGetter(abc.ABC):
    @abc.abstractmethod
    def get_state(self, simulation: Simulation) -> list[np.ndarray]:
        pass


class FullJointStateGetter(CreatureStateGetter):
    def get_state(self, simulation: Simulation)  -> npt.NDArray[np.float64]:
        r_positions, r_velocities = simulation.get_revolute_joint_states()
        s_positions, s_velocities = simulation.get_spherical_joint_states()

        return _combine_inputs([r_positions, r_velocities, s_positions, s_velocities])


def _combine_inputs(arrays) -> npt.NDArray:
    """
    Takes N inputs (arrays or lists), flattens them, 
    and combines them into a single 1D NumPy array.
    """
    processed_list = [np.asanyarray(x).ravel() for x in arrays]
    
    return np.concatenate(processed_list)


