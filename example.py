import pybullet as p
import pybullet_data
import time
import random

#
#  a temporary test file
#

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, 0)


planeId = p.loadURDF("plane.urdf")
temp_start_pos = [0, 0, 0]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])
creatureId = p.loadURDF("assets/creatures/4-spherical_4-revolute_spider-bot.urdf", 
                        temp_start_pos, 
                        start_orientation)

p.stepSimulation()


global_min_z = p.getAABB(creatureId, -1)[0][2] 
num_links = p.getNumJoints(creatureId)
for i in range(num_links):
    global_min_z = min(global_min_z, p.getAABB(creatureId, i)[0][2])


buffer = 0.001 
spawn_z = -global_min_z + buffer
start_position = [0, 0, spawn_z]
p.resetBasePositionAndOrientation(creatureId, 
                                start_position, 
                                start_orientation)


p.setGravity(0, 0, -9.81)

settle_steps = 120 
print(f"Settling creature for {settle_steps} steps...")
for _ in range(settle_steps):
    p.stepSimulation()
    if physicsClient == p.GUI:
        time.sleep(1./240.) 


print("Settling complete. Saving state.")
startStateId = p.saveState()

single_dof_joints = []
spherical_joints = []

for i in range(p.getNumJoints(creatureId)):
    info = p.getJointInfo(creatureId, i)
    joint_type = info[2] 

    if joint_type == p.JOINT_SPHERICAL:
        spherical_joints.append(i)
    elif joint_type != p.JOINT_FIXED:
        single_dof_joints.append(i)

print(f"Controlling {len(single_dof_joints)} single-DoF joints: {single_dof_joints}")
print(f"Controlling {len(spherical_joints)} spherical joints: {spherical_joints}")

p.setRealTimeSimulation(0)
for i in range(10000):
    if i % 1000 == 0:
        print(f"Step {i}: Resetting to start state.")
        p.restoreState(startStateId)
        
        for _ in range(60): 
            p.stepSimulation()
            time.sleep(1./240.)

    for joint_index in single_dof_joints:
        max_force = 100.0
        torque = random.uniform(-max_force, max_force)
        
        p.setJointMotorControl2(
            bodyUniqueId=creatureId,
            jointIndex=joint_index,
            controlMode=p.TORQUE_CONTROL,
            force=torque
        )
        
    for joint_index in spherical_joints:
        max_force = 100.0 
        torque_vector = [
            random.uniform(-max_force, max_force), 
            random.uniform(-max_force, max_force), 
            random.uniform(-max_force, max_force)  
        ]
        
        p.setJointMotorControlMultiDof(
            bodyUniqueId=creatureId,
            jointIndex=joint_index,
            controlMode=p.TORQUE_CONTROL,
            force=torque_vector 
        )

    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()