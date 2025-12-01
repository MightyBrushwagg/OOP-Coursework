import pybullet as p
import pybullet_data
import time
import math
import os
from collections import namedtuple

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


# ---------- Load Free-Flying Gripper ----------
gripper_start_pos = [0,0,1]
gripper_start_ori = p.getQuaternionFromEuler([math.pi,0,0])
gripper_id = p.loadURDF(os.path.join(script_dir, "robotiq_2f_85", "robotiq.urdf"), 
                         gripper_start_pos, gripper_start_ori, useFixedBase=False)

# ---------- Parse joints ----------
num_joints = p.getNumJoints(gripper_id)
JointInfo = namedtuple('JointInfo',['id','name','type','lower','upper','maxForce'])
joints = []
for i in range(num_joints):
    info = p.getJointInfo(gripper_id, i)
    jid = info[0]
    name = info[1].decode()
    jtype = info[2]
    lower = info[8]
    upper = info[9]
    maxForce = info[10]
    joints.append(JointInfo(jid,name,jtype,lower,upper,maxForce))
    p.setJointMotorControl2(gripper_id,jid,p.VELOCITY_CONTROL,targetVelocity=0,force=0)

# ---------- Mimic Joint Setup ----------
mimic_parent_name = 'finger_joint'
mimic_children_names = {'right_outer_knuckle_joint':1,
                        'left_inner_knuckle_joint':1,
                        'right_inner_knuckle_joint':1,
                        'left_inner_finger_joint':-1,
                        'right_inner_finger_joint':-1}

mimic_parent_id = [j.id for j in joints if j.name==mimic_parent_name][0]
mimic_child_multiplier = {j.id: mimic_children_names[j.name] for j in joints if j.name in mimic_children_names}

# Create mimic constraints
for joint_id, multiplier in mimic_child_multiplier.items():
    c = p.createConstraint(gripper_id,mimic_parent_id,
                           gripper_id,joint_id,
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-multiplier,maxForce=100,erp=1)

gripper_range = [0,0.085]  # min open, max open

def move_gripper(open_length):
    open_angle = 0.715 - math.asin((open_length-0.010)/0.1143)
    p.setJointMotorControl2(gripper_id,mimic_parent_id,p.POSITION_CONTROL,
                            targetPosition=open_angle, force=60)  # increase force

def set_gripper_position(pos, use_velocity=False):
    """
    Move gripper to position. 
    If use_velocity=True, uses velocity control to avoid clipping.
    Otherwise uses position reset (only safe before grasping).
    """
    if use_velocity:
        # Get current position
        current_pos, current_ori = p.getBasePositionAndOrientation(gripper_id)
        # Calculate velocity needed
        velocity = [(pos[i] - current_pos[i]) * 10 for i in range(3)]  # proportional control
        p.resetBaseVelocity(gripper_id, linearVelocity=velocity)
    else:
        p.resetBasePositionAndOrientation(gripper_id, pos, gripper_start_ori)

def grasp_and_lift(obj_id, obj_pos, lift_height=0.4, approach_steps=60, lift_steps=150, hold_time=10.0):
    """
    Grasp an object and lift it to a target height, then hold it.
    
    Args:
        obj_id: PyBullet object ID to grasp
        obj_pos: [x, y, z] position of the object
        lift_height: Target height to lift to (relative to grasp position)
        approach_steps: Number of steps for smooth descent to grasp position
        lift_steps: Number of simulation steps for lifting
        hold_time: Time (in seconds) to hold the object at lifted height
    """
    # Height of gripper base when fingers are at object height
    # Gripper is upside down, so base needs to be ~12cm above object to align fingers
    # Adding extra 0.5cm to prevent initial clipping
    grasp_height = obj_pos[2] + 0.125
    time_step = 1./240.
    
    # --- Set friction on contact surfaces ---
    # Set MAXIMUM friction on object (all links)
    for i in range(-1, p.getNumJoints(obj_id)):
        p.changeDynamics(obj_id, i, 
                        lateralFriction=20.0,      # Very high friction
                        rollingFriction=0.5,       # Prevent rolling
                        spinningFriction=0.5,      # Prevent spinning
                        contactStiffness=10000,    # Very stiff contact
                        contactDamping=2000)       # High damping
    
    # Set MAXIMUM friction on ALL gripper links (especially finger tips)
    for i in range(-1, p.getNumJoints(gripper_id)):
        p.changeDynamics(gripper_id, i, 
                        lateralFriction=20.0,      # Very high friction
                        rollingFriction=0.5,       # Prevent rolling
                        spinningFriction=0.5,      # Prevent spinning
                        contactStiffness=10000,    # Very stiff contact
                        contactDamping=2000)       # High damping

    # Start position - well above the object
    approach_height = grasp_height + 0.15
    
    # Open gripper fully before approaching
    move_gripper(gripper_range[1])
    
    # Smoothly move gripper down from start position to approach height
    # Use controlled velocity to prevent gravity-induced acceleration
    z_start = gripper_start_pos[2]
    descent_velocity = (approach_height - z_start) / (approach_steps * time_step)
    
    for step in range(approach_steps):
        z_current = z_start + descent_velocity * time_step * step
        pos = [obj_pos[0], obj_pos[1], z_current]
        set_gripper_position(pos)
        # Set controlled velocity to counteract gravity
        p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, descent_velocity], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(time_step)
    
    # No pause - continue smoothly to grasp position

    # Smoothly lower to grasp height - faster descent
    descent_steps = 40
    descent_velocity = (grasp_height - approach_height) / (descent_steps * time_step)
    
    for step in range(descent_steps):
        z_current = approach_height + descent_velocity * time_step * step
        pos = [obj_pos[0], obj_pos[1], z_current]
        set_gripper_position(pos)
        # Set controlled velocity to counteract gravity
        p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, descent_velocity], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(time_step)

    # Brief stabilization at grasp position
    p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    for _ in range(10):
        p.stepSimulation()
        time.sleep(time_step)

    # --- Close gripper gently around object ---
    # Don't close fully - leave small gap to avoid clipping
    # Smaller gap = better contact without crushing
    target_close_position = gripper_range[0] + 0.001  # Leave 1mm gap
    open_angle = 0.715 - math.asin((target_close_position-0.010)/0.1143)
    
    # Close with MAXIMUM force and slow speed for firm grip
    print("Closing gripper with maximum force...")
    p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                            targetPosition=open_angle, force=2000, maxVelocity=0.5)
    
    # Give plenty of time for firm contact to establish
    for i in range(200):  # Longer closing time for better grip
        # Reset velocity every 10 steps to prevent drift
        if i % 10 == 0:
            p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        p.stepSimulation()
        time.sleep(time_step)

    # --- Lift with continuous firm grip ---
    print(f"Starting lift phase: {lift_steps} steps, target height: {lift_height}m")
    # Use constant upward velocity with maximum grip force
    lift_velocity = lift_height / (lift_steps * time_step)
    print(f"Lift velocity: {lift_velocity:.4f} m/s")
    
    for step in range(lift_steps):
        # Set constant upward velocity
        p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, lift_velocity], angularVelocity=[0, 0, 0])
        
        # MAXIMUM grip force to prevent slipping - this is the key!
        p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                                targetPosition=open_angle, force=2000, maxVelocity=0.5)

        p.stepSimulation()
        time.sleep(time_step)
        
        # Progress indicator
        if step % 100 == 0 and step > 0:
            progress_pct = (step / lift_steps) * 100
            current_pos, _ = p.getBasePositionAndOrientation(gripper_id)
            print(f"  Lifting: {progress_pct:.1f}% complete, height: {current_pos[2]:.3f}m")
    
    # Stop upward motion
    p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
    
    # --- Hold phase - maintain grip and position to observe stability ---
    print(f"Holding object at lifted height for {hold_time} seconds to check grip stability...")
    hold_steps = int(hold_time / time_step)
    
    # Get final position to maintain
    final_pos, _ = p.getBasePositionAndOrientation(gripper_id)
    hold_height = final_pos[2]
    
    for step in range(hold_steps):
        # Get current position
        current_pos, _ = p.getBasePositionAndOrientation(gripper_id)
        
        # Apply small corrective velocity if drifting
        height_error = hold_height - current_pos[2]
        corrective_velocity = height_error * 2.0  # Gentle proportional control
        
        # Set velocity to counteract any drift
        p.resetBaseVelocity(gripper_id, linearVelocity=[0, 0, corrective_velocity], angularVelocity=[0, 0, 0])
        
        # MAXIMUM grip force to prevent slipping
        p.setJointMotorControl2(gripper_id, mimic_parent_id, p.POSITION_CONTROL,
                                targetPosition=open_angle, force=2000, maxVelocity=0.5)
        
        p.stepSimulation()
        time.sleep(time_step)
        
        # Print status every second
        if step % int(1.0 / time_step) == 0:
            elapsed = step * time_step
            remaining = hold_time - elapsed
            print(f"  Holding... {remaining:.1f}s remaining, height: {current_pos[2]:.3f}m")

# ---------- Pick-Up Sequence ----------

# Execute grasp and lift with extended hold time
print("Starting grasp and lift sequence...")
print(f"Target: Lift cube from {box_pos[2]:.3f}m to {box_pos[2] + 0.3:.3f}m")
grasp_and_lift(box_id, box_pos, lift_height=0.3, lift_steps=200, hold_time=15.0)

# Keep GUI open for additional observation
print("\nGrasp and lift complete! Check if the object stayed gripped.")
print("Keeping simulation window open for final observation...")
time.sleep(10)
p.disconnect()
