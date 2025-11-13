import pybullet as p
import pybullet_data
import time
import random


physicsClient = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

start_position = [0, 0, 0.5] 
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
creatureId = p.loadURDF("biped/biped2d_pybullet.urdf", start_position, start_orientation)

joint_indices = []
for i in range(p.getNumJoints(creatureId)):
    if p.getJointInfo(creatureId, i)[2] != p.JOINT_FIXED:
        joint_indices.append(i)

print(f"Controlling {len(joint_indices)} joints: {joint_indices}")

p.setRealTimeSimulation(0) 
for _ in range(10000):
    for joint_index in joint_indices:
        max_force = 350.0 
        torque = random.uniform(-max_force, max_force) 
        
        print(torque)
        p.setJointMotorControl2(
            bodyUniqueId=creatureId,
            jointIndex=joint_index,
            controlMode=p.TORQUE_CONTROL,
            force=torque
        )
    
    p.stepSimulation()
    time.sleep(1./240.) 

p.disconnect()