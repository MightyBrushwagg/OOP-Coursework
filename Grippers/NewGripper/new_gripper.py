import pybullet as p
import pybullet_data
import time
import math
import os
from collections import namedtuple

# ========== CONSTANTS ==========
TIME_STEP = 1.0 / 240.0
GRIPPER_OFFSET = 0.125  # Height offset from object to gripper base
APPROACH_OFFSET = 0.15  # Extra height for approach phase
CLOSE_GAP = 0.001  # Gap left when closing (meters)
MAX_GRIP_FORCE = 2000
GRIP_VELOCITY = 0.5

# Friction and contact parameters
FRICTION_PARAMS = {
    'lateralFriction': 20.0,
    'rollingFriction': 0.5,
    'spinningFriction': 0.5,
    'contactStiffness': 10000,
    'contactDamping': 2000
}

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# ---------- Setup ----------
p.connect(p.GUI)
# Add search paths so mesh files can be found
p.setAdditionalSearchPath(parent_dir)
p.setAdditionalSearchPath(os.path.join(script_dir, "robotiq_2f_85"))
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0,0,-10)
plane_id = p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=40,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# ---------- Load Box ----------
box_pos = [0, 0, 0.03]
box_id = p.loadURDF(os.path.join(script_dir, "cube_small.urdf"), box_pos)


# ---------- Load Gripper with Fixed Constraint ----------
gripper_start_pos = [0,0,1]
gripper_start_ori = p.getQuaternionFromEuler([math.pi,0,0])
gripper_id = p.loadURDF(os.path.join(script_dir, "robotiq_2f_85", "robotiq.urdf"), 
                         gripper_start_pos, gripper_start_ori, useFixedBase=False)

# Create a fixed constraint to hold gripper in place (like simulation.py does)
constraint_id = p.createConstraint(
    parentBodyUniqueId=gripper_id,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=gripper_start_pos,
    childFrameOrientation=gripper_start_ori
)

# ---------- Parse and disable default joint motors ----------
JointInfo = namedtuple('JointInfo', ['id', 'name', 'type', 'lower', 'upper', 'maxForce'])
joints = []

for i in range(p.getNumJoints(gripper_id)):
    info = p.getJointInfo(gripper_id, i)
    joints.append(JointInfo(info[0], info[1].decode(), info[2], info[8], info[9], info[10]))
    p.setJointMotorControl2(gripper_id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

# ---------- Setup mimic joints (link fingers together) ----------
mimic_parent_id = next(j.id for j in joints if j.name == 'finger_joint')
mimic_children = {
    'right_outer_knuckle_joint': 1, 'left_inner_knuckle_joint': 1,
    'right_inner_knuckle_joint': 1, 'left_inner_finger_joint': -1,
    'right_inner_finger_joint': -1
}

for joint in joints:
    if joint.name in mimic_children:
        c = p.createConstraint(gripper_id, mimic_parent_id, gripper_id, joint.id,
                               jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-mimic_children[joint.name], maxForce=100, erp=1)

gripper_range = [0, 0.085]  # min open, max open

def step_simulation():
    """Single simulation step with proper timing."""
    p.stepSimulation()
    time.sleep(TIME_STEP)

def calc_gripper_angle(open_length):
    """Calculate gripper joint angle from opening length."""
    return 0.715 - math.asin((open_length - 0.010) / 0.1143)

def move_gripper(open_length, force=60, max_velocity=None):
    """Move gripper to specified opening."""
    open_angle = calc_gripper_angle(open_length)
    if max_velocity is not None:
        p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                                targetPosition=open_angle, force=force, maxVelocity=max_velocity)
    else:
        p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                                targetPosition=open_angle, force=force)

def set_gripper_position(pos, orientation=None):
    """Move gripper to position using constraint."""
    if orientation is None:
        orientation = gripper_start_ori
    p.changeConstraint(constraint_id, jointChildPivot=pos,
                      jointChildFrameOrientation=orientation, maxForce=500)

def set_high_friction(body_id):
    """Apply maximum friction to all links of a body."""
    for i in range(-1, p.getNumJoints(body_id)):
        p.changeDynamics(body_id, i, **FRICTION_PARAMS)

def move_to_height(x, y, z_start, z_end, steps):
    """Smoothly move gripper from z_start to z_end height."""
    z_step = (z_end - z_start) / steps
    for step in range(steps):
        z_current = z_start + z_step * step
        set_gripper_position([x, y, z_current])
        step_simulation()

def grasp_and_lift(obj_id, obj_pos, lift_height=0.4, approach_steps=60, lift_steps=150, hold_time=10.0):
    """Grasp an object and lift it to a target height, then hold it."""
    # Calculate key heights
    grasp_height = obj_pos[2] + GRIPPER_OFFSET
    approach_height = grasp_height + APPROACH_OFFSET
    x, y = obj_pos[0], obj_pos[1]
    
    # Apply maximum friction to prevent slipping
    set_high_friction(obj_id)
    set_high_friction(gripper_id)
    
    # Open gripper and descend to approach position
    move_gripper(gripper_range[1])
    move_to_height(x, y, gripper_start_pos[2], approach_height, approach_steps)
    
    # Lower to grasp height
    move_to_height(x, y, approach_height, grasp_height, 40)
    
    # Brief stabilization
    for _ in range(10):
        step_simulation()
    
    # Close gripper around object
    print("Closing gripper...")
    close_angle = calc_gripper_angle(gripper_range[0] + CLOSE_GAP)
    move_gripper(gripper_range[0] + CLOSE_GAP, force=MAX_GRIP_FORCE, max_velocity=GRIP_VELOCITY)
    for _ in range(200):
        step_simulation()
    
    # Lift with continuous grip
    print(f"Lifting {lift_height}m...")
    z_current = grasp_height
    z_step = lift_height / lift_steps
    
    for step in range(lift_steps):
        z_current += z_step
        set_gripper_position([x, y, z_current])
        p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                                targetPosition=close_angle, force=MAX_GRIP_FORCE, maxVelocity=GRIP_VELOCITY)
        step_simulation()
        
        if step % 100 == 0 and step > 0:
            pos, _ = p.getBasePositionAndOrientation(gripper_id)
            print(f"  {100*step//lift_steps}% complete, height: {pos[2]:.3f}m")
    
    # Hold and observe
    print(f"Holding for {hold_time}s...")
    hold_steps = int(hold_time / TIME_STEP)
    
    for step in range(hold_steps):
        p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                                targetPosition=close_angle, force=MAX_GRIP_FORCE, maxVelocity=GRIP_VELOCITY)
        step_simulation()
        
        if step % int(1.0 / TIME_STEP) == 0:
            remaining = hold_time - step * TIME_STEP
            pos, _ = p.getBasePositionAndOrientation(gripper_id)
            print(f"  {remaining:.1f}s remaining, height: {pos[2]:.3f}m")

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("=== Grasp and Lift Sequence ===")
    print(f"Target: Lift cube from {box_pos[2]:.3f}m to {box_pos[2] + 0.3:.3f}m\n")
    
    grasp_and_lift(box_id, box_pos, lift_height=0.3, lift_steps=200, hold_time=15.0)
    
    print("\n=== Complete! ===")
    print("Keeping simulation open for 10 seconds...")
    time.sleep(10)
    
    # Cleanup
    p.removeConstraint(constraint_id)
    p.disconnect()
