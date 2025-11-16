import pybullet as p
import pybullet_data
from pathlib import Path
import numpy as np
import numpy.typing as npt
import time

class Simulation:
    """The Simulation class is an environment for running consecutive creature walking simulations optimally.
    It is suited for simulating one creature at a time. The reset_state method should be called after each individual run.
    For parallel processing, separate Simulation instances should be used.
    """
    def __init__(self, simulation_type: int, creature_path: Path, settle_steps: int = 120, time_step: float = 1./240.):
        """Initializes the Simulation class.

            :param int simulation_type: If the simulation should include a graphical representation (p.GUI)
                                            or not (p.DIRECT). Use p.DIRECT for performance.
            :param Path creature_path: A path to the .urdf creature file that should be used.
            :param int, optional settle_steps: How many simulation steps to wait for the creature to reach a stable state.
                                            Probably not necessary to change the default. Defaults to 120.
            :param float, optional time_step: How much time should one tick represent (NOT how much real time should it take). Defaults to 1./240..
        """
        if simulation_type not in (p.GUI, p.DIRECT):
            raise ValueError(f"Invalid simulation_type. "
                            f"Must be p.GUI or p.DIRECT, but got {simulation_type}")
        
        self.gui_enabled = (simulation_type == p.GUI)
        self.time_step = time_step

        self.client_id = p.connect(simulation_type) 
        
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        # enable access to default pybullet assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        # turn gravity off to measure the creature's bounding box
        p.setGravity(0, 0, 0, physicsClientId=self.client_id)

        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        
        # load creature at origin (temp)
        start_orientation = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.client_id)
        temp_start_pos = [0, 0, 0]
        self.creatureId = p.loadURDF(str(creature_path), 
                                    temp_start_pos, 
                                    start_orientation, 
                                    physicsClientId=self.client_id)

        # perform one simulation step to register all link AABBs
        p.stepSimulation(physicsClientId=self.client_id)

        # find the true lowest point by checking all links
        global_min_z = p.getAABB(self.creatureId, -1, physicsClientId=self.client_id)[0][2]
        num_links = p.getNumJoints(self.creatureId, physicsClientId=self.client_id)
        for i in range(num_links):
            link_aabb = p.getAABB(self.creatureId, i, physicsClientId=self.client_id)
            global_min_z = min(global_min_z, link_aabb[0][2])

        # calculate spawn height and teleport
        buffer = 0.001 # a buffer to ensure the creature doesn't clip into the ground
        spawn_z = -global_min_z + buffer
        start_position = [0, 0, spawn_z]
        p.resetBasePositionAndOrientation(self.creatureId, 
                                        start_position, 
                                        start_orientation, 
                                        physicsClientId=self.client_id)

        # turn gravity on and let the creature settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        
        # find the creature's stable pose.
        for _ in range(settle_steps):
            p.stepSimulation(physicsClientId=self.client_id)

        # save the stable starting state to use after resets
        self.startStateId = p.saveState(physicsClientId=self.client_id)
        
        revolute_joints_list = []
        spherical_joints_list = []
        for i in range(p.getNumJoints(self.creatureId, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.creatureId, i, physicsClientId=self.client_id)
            joint_type = info[2] 

            if joint_type == p.JOINT_SPHERICAL:
                spherical_joints_list.append(i)
            elif joint_type != p.JOINT_FIXED:
                revolute_joints_list.append(i)
        
        self.revolute_joints: npt.NDArray[np.int32] = np.array(revolute_joints_list, dtype=np.int32)
        self.spherical_joints: npt.NDArray[np.int32] = np.array(spherical_joints_list, dtype=np.int32)

        self.num_revolute = len(self.revolute_joints)
        self.num_spherical = len(self.spherical_joints)
        
        # initialize the tick count to measure how many ticks passed
        self.tick_count = 0

    
    def step(self):
        """Advances the simulation by one physics step.
        Call this after applying all desired joint movements for that step.
        """
        p.stepSimulation(physicsClientId=self.client_id)
        
        self.tick_count += 1
        
        if self.gui_enabled:
            time.sleep(self.time_step)
    
    
    def get_creature_position(self) -> npt.NDArray[np.float64]:
        """
        Gets the position of the creature's base (torso).
        
        :return npt.NDArray[np.float64]: A 1D NumPy array [x, y, z] of the base position.
        """
        pos, _ = p.getBasePositionAndOrientation(self.creatureId, physicsClientId=self.client_id)
        return np.array(pos, dtype=np.float64)
    
    
    def get_tick_count(self) -> int:
        """Returns the tick count since the start of the current run. The tick count can be thought of as the 'time passed'.
        It is reset every time reset_state is called.

        :return int: An int representing the 'time passed' since the start of the run.
        """
        return self.tick_count
    
    
    def moveRevolute(self, joint_ids: npt.NDArray[np.int32], torques: npt.NDArray[np.float64]):
        """
        Applies torques to a specified list of revolute (1-DOF) joints.
        
        :param npt.NDArray[np.int32] joint_ids: A 1D NumPy array of joint indices.
        :param npt.NDArray[np.float64] torques: A 1D NumPy array of torque values. Must have the same
                        length as joint_ids.
        """
        if joint_ids.shape[0] == 0:
            return
            
        if joint_ids.shape[0] != torques.shape[0]:
            raise ValueError(f"joint_ids and torques arrays must have the same length. "
                            f"Got {joint_ids.shape[0]} and {torques.shape[0]}")
        
        p.setJointMotorControlArray(
            bodyUniqueId=self.creatureId,
            jointIndices=joint_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.client_id
        )
    
    
    def moveSpherical(self, joint_ids: npt.NDArray[np.int32], torques: npt.NDArray[np.float64]):
        """
        Applies torques to a specified list of spherical (3-DOF) joints.
        
        :param npt.NDArray[np.int32] joint_ids: A 1D NumPy array of joint indices.
        :param npt.NDArray[np.float64] torques: A 2D NumPy array of torque vectors. Must have shape
                        (len(joint_ids), 3).
        """
        if joint_ids.shape[0] == 0:
            return

        if joint_ids.shape[0] != torques.shape[0]:
            raise ValueError(f"joint_ids and torques arrays must have the same length (dim 0). "
                            f"Got {joint_ids.shape[0]} and {torques.shape[0]}")

        if torques.ndim != 2 or torques.shape[1] != 3:
            raise ValueError(f"Torques array has wrong shape. "
                            f"Expected (N, 3) but got {torques.shape}")
        
        torques_flat = torques.flatten()
        
        p.setJointMotorControlMultiDofArray(
            bodyUniqueId=self.creatureId,
            jointIndices=joint_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=torques_flat,
            physicsClientId=self.client_id
        )
    
    
    def reset_state(self):
        """Resets the simulation to the saved 'settled' start state.
        Call this after having finished the simulation for a single individual."""
        p.restoreState(self.startStateId, physicsClientId=self.client_id)
        self.tick_count = 0


    def terminate(self):
        """Disconnects from the PyBullet physics server.
        Call this after finishing all necessary simulations using this object."""
        p.disconnect(physicsClientId=self.client_id)