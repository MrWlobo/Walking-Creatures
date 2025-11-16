import pybullet as p
import pybullet_data
from pathlib import Path

class Simulation:
    def __init__(self, simulation_type: int, creature_path: Path, settle_steps: int = 120):
        if simulation_type not in (p.GUI, p.DIRECT):
            raise ValueError(f"Invalid simulation_type. "
                            f"Must be p.GUI or p.DIRECT, but got {simulation_type}")
        
        self.gui_enabled = (simulation_type == p.GUI)

        self.physicsClient = p.connect(simulation_type) 

        # enable access to default pybullet assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # turn gravity off to measure the creature's bounding box
        p.setGravity(0, 0, 0)

        # load plane and creature at origin (temp)
        self.planeId = p.loadURDF("plane.urdf")
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        temp_start_pos = [0, 0, 0]
        self.creatureId = p.loadURDF(str(creature_path), 
                                    temp_start_pos, 
                                    self.start_orientation)

        # perform one simulation step to register all link AABBs
        p.stepSimulation()

        # find the true lowest point by checking all links
        global_min_z = p.getAABB(self.creatureId, -1)[0][2]
        num_links = p.getNumJoints(self.creatureId)
        for i in range(num_links):
            global_min_z = min(global_min_z, p.getAABB(self.creatureId, i)[0][2])

        # calculate spawn height and teleport
        buffer = 0.001 # a buffer to ensure the creature doesn't clip into the ground
        spawn_z = -global_min_z + buffer
        self.start_position = [0, 0, spawn_z]
        p.resetBasePositionAndOrientation(self.creatureId, 
                                        self.start_position, 
                                        self.start_orientation)

        # turrn gravity on and let the creature settle
        p.setGravity(0, 0, -9.81)
        
        # find the creature's stable pose.
        for _ in range(settle_steps):
            p.stepSimulation()

        # save the stable state to use after resets
        self.startStateId = p.saveState()


    def reset_state(self):
        p.restoreState(self.startStateId)


    def terminate(self):
        p.disconnect()