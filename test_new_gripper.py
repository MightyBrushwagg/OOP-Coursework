"""Test NewGripper class with simulation.py structure"""
import pybullet as p
import pybullet_data
import time
from objects import Box
from grippers import NewGripper

# Setup simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)
plane_id = p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=40,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# Create object
obj = Box([0, 0, 0.075])
obj.load()
obj.update_name(obj.id)

# Create gripper (upside down like in new_gripper.py)
gripper = NewGripper(base_position=[0, 0, 1], orientation=[3.14159, 0, 0])
gripper.load()
gripper.start()

# Attach fixed constraint (like simulation.py does)
gripper.attach_fixed(offset=[0, 0, 0])

# Execute grasp and lift
print("=== Testing NewGripper with simulation.py structure ===")
gripper.grasp_and_lift(obj, lift_height=0.3, lift_steps=200, hold_time=10.0)

print("\n=== Test Complete ===")
time.sleep(5)

# Cleanup
p.removeConstraint(gripper.constraint_id)
p.removeBody(gripper.id)
p.removeBody(obj.id)
p.disconnect()

