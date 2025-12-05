import pybullet as p
import pybullet_data
import time
from abc import ABC, abstractmethod
import numpy as np
import os
from collections import namedtuple
import math

class Gripper(ABC):
    """Base gripper class that defines common gripper behavior."""
    def __init__(self, urdf_path, base_position, orientation=(0, 0, 0), visuals="visuals"):
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.id = None
        self.constraint_id = None
        self.grasp_moving = False
        self.orientation = orientation
        self.visuals = visuals
        if self.visuals == "no visuals":
            self.timestep = 0
        else:
            self.timestep = 1./240.

    def load(self):
        """Load gripper into the PyBullet world."""
        self.id = p.loadURDF(self.urdf_path, self.base_position, p.getQuaternionFromEuler(self.orientation))
        return self.id
    
    def attach_fixed(self, offset):
        """Attach gripper to a fixed world position."""
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.base_position,
            childFrameOrientation=p.getQuaternionFromEuler(self.orientation)
        )

    def move(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Move gripper to a new position and orientation."""
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation=p.getQuaternionFromEuler([roll, pitch, yaw]),
            maxForce=100
        )
 
    def update_camera(self, z, yaw):
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=50 + (yaw * 180 / 3.1416),
            cameraPitch=-60,
            cameraTargetPosition=[0.5, 0.3, z]
        )


class TwoFingerGripper(Gripper):
    """A specific two-finger gripper inheriting from Gripper class.
    Initialize the gripper using default base_position=(0.5, 0.3, 0.7) and the urdf name: pr2_gripper.urdf setting up the relevant attributes.
    You need to use the parent's initializer accordingly.
    """ 
    def __init__(self, base_position=(0,0,0), orientation=(0, 0, 0), visuals="visuals"):
       # Setup the gripper
        super().__init__("pr2_gripper.urdf", base_position=base_position, orientation=orientation, visuals=visuals)

    def start(self):
        """Open gripper at start."""
        initial_positions = [0.550569, 0.0, 0.549657, 0.0]
        for i, pos in enumerate(initial_positions):
            p.resetJointState(self.id, i, pos)

    def open(self):
        """Open the gripper fingers."""
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, maxVelocity=1, force=10)

    def close(self):
        """Close the gripper fingers."""
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.1, maxVelocity=1, force=10)
    
    def generate_angles(self, gripper_pos, object_pos):
        direction = np.array(object_pos) - np.array(gripper_pos)
        direction = direction/np.linalg.norm(direction)

        pitch = np.arcsin(-direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        roll = np.pi

        return np.array([roll, pitch, yaw])

    def target_position(self, obj):
        approach_pos = [self.base_position[0], self.base_position[1], self.base_position[2] + 0.1]
        off = 0.04 # Need to iterate on this value
        direction = np.array(approach_pos) - np.array(obj.position)
        direction[2] = 0 # No z term offset needed because offset doesn't change based on starting position 

        if np.linalg.norm(direction) > 0: # No zero lengths 
            direction = direction / np.linalg.norm(direction) # Normalize direction vector
            offset = direction * off # Scale by offset
        else:
            offset = np.array([0, 0, 0])

        target_pos = np.array(obj.position) + offset # Apply offset
        return target_pos, approach_pos

    def grasp_and_lift(self, obj, lift_height=0.4, lift_steps=150):
        """Perform grasping and lifting sequence with sustained gripping force."""
        target_pos, approach_pos = self.target_position(obj)
        roll, pitch, yaw = self.generate_angles(approach_pos, target_pos) # Generate angles based on offset obj pos
        # roll = self.orientation[0]
        # pitch = self.orientation[1]
        # yaw = self.orientation[2]
        grasp_height = obj.grasp_height

        # --- Set friction on contact surfaces ---
        p.changeDynamics(obj.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)

        # --- Move above object by calling the move function inherited from the parent
        self.move(target_pos[0], target_pos[1], target_pos[2] + 0.2, roll, pitch, yaw)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.timestep)

        # --- Lower onto object ---
        self.move(target_pos[0], target_pos[1], grasp_height, roll, pitch, yaw)
        # print("\033[93mmove to the grasp height")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.timestep)

        # --- Close gripper strongly ---
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.12, force=300, maxVelocity=2)
        for _ in range(100):  # allow contact to form
            p.stepSimulation()
            time.sleep(self.timestep)

        # --- Continuous hold while lifting step-by-step ---
        z_current = grasp_height
        z_target = lift_height
        z_step = (z_target - z_current) / lift_steps

        for _ in range(lift_steps):
            z_current += z_step
            self.move(target_pos[0], target_pos[1], z_current, roll, pitch, yaw)

            # Continuously reapply strong grip to prevent slip
            for joint in [0, 2]:
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                        targetPosition=0.12, force=400, maxVelocity=2)

            p.stepSimulation()
            time.sleep(self.timestep)

class NewGripper(Gripper):
    def __init__(self, base_position=(0,0,0), orientation=(0, 0, 0), visuals="visuals"):
        # Get the correct URDF path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "NewGripper", "robotiq_2f_85", "robotiq.urdf")
        super().__init__(urdf_path, base_position=base_position, orientation=orientation, visuals=visuals)
        self.mimic_parent_id = None
        self.mimic_child_multiplier = {}
        self.gripper_range = [0, 0.085]  # min open, max open

    def start(self):
        """Initialize gripper joints and setup mimic joint constraints."""
        # Parse joints
        num_joints = p.getNumJoints(self.id)
        JointInfo = namedtuple('JointInfo', ['id', 'name', 'type', 'lower', 'upper', 'maxForce'])
        joints = []
        
        for i in range(num_joints):
            info = p.getJointInfo(self.id, i)
            jid = info[0]
            name = info[1].decode()
            jtype = info[2]
            lower = info[8]
            upper = info[9]
            maxForce = info[10]
            joints.append(JointInfo(jid, name, jtype, lower, upper, maxForce))
            # Disable default motor control
            p.setJointMotorControl2(self.id, jid, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        
        # Setup mimic joints
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        
        self.mimic_parent_id = [j.id for j in joints if j.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {j.id: mimic_children_names[j.name] 
                                       for j in joints if j.name in mimic_children_names}
        
        # Create mimic constraints
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id, self.mimic_parent_id,
                self.id, joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)
        
        self.open()
    
    def calc_gripper_angle(self, open_length):
        """Calculate gripper joint angle from opening length."""
        return 0.715 - math.asin((open_length - 0.010) / 0.1143)
    
    def move_gripper(self, open_length, force=100, max_velocity=None):
        """Move gripper to specified opening length."""
        open_angle = self.calc_gripper_angle(open_length)
        if max_velocity is not None:
            p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL,
                                    targetPosition=open_angle, force=force, maxVelocity=max_velocity)
        else:
            p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL,
                                    targetPosition=open_angle, force=force)
    
    def open(self):
        """Open the gripper fully."""
        self.move_gripper(self.gripper_range[1])
    
    def close(self, force=2000, max_velocity=0.5):
        """Close the gripper with specified force."""
        target_close_position = self.gripper_range[0] + 0.001  # Leave 1mm gap
        self.move_gripper(target_close_position, force=force, max_velocity=max_velocity)

    def generate_angles(self, gripper_pos, object_pos):
        direction = np.array(object_pos) - np.array(gripper_pos)
        direction = direction/np.linalg.norm(direction)

        pitch = np.arcsin(-direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        roll = np.pi

        return np.array([roll, -pitch, yaw])
    
    def target_position(self, obj):
        approach_pos = [self.base_position[0] - 0.2, self.base_position[1], self.base_position[2] + 0.1]
        off = -0.12 # Need to iterate on this value
        direction = np.array(approach_pos) - np.array(obj.position)
        direction[2] = 0 # No z term offset needed because offset doesn't change based on starting position 

        if np.linalg.norm(direction) > 0: # No zero lengths 
            direction = direction / np.linalg.norm(direction) # Normalize direction vector
            offset = direction * off # Scale by offset
        else:
            offset = np.array([0, 0, 0])

        target_pos = np.array(obj.position) + offset # Apply offset
        return target_pos, approach_pos
    
    def grasp_and_lift(self, obj, lift_height=0.4, lift_steps=150):
        """Perform grasping and lifting sequence with sustained gripping force."""
        target_pos, approach_pos = self.target_position(obj)
        
        # Calculate angles to point gripper from approach position towards target (like TwoFingerGripper)
        roll, pitch, yaw = self.generate_angles(approach_pos, target_pos)
       
        grasp_height = obj.grasp_height + 0.2
        
        # --- Set friction on contact surfaces ---
        # Keep high friction values for NewGripper (they work better)
        friction_params = {
            'lateralFriction': 20.0, 'rollingFriction': 0.5, 'spinningFriction': 0.5,
            'contactStiffness': 10000, 'contactDamping': 2000
        }
        for i in range(-1, p.getNumJoints(obj.id)):
            p.changeDynamics(obj.id, i, **friction_params)
        for i in range(-1, p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, **friction_params)
        
        # --- Move above object by calling the move function inherited from the parent
        self.move(target_pos[0], target_pos[1], target_pos[2] + 0.3, roll, pitch, yaw)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.timestep)
        
        # --- Lower onto object ---
        # Use target_pos (object x,y) with calculated grasp_height (z)
        self.move(target_pos[0], target_pos[1], grasp_height, roll, pitch, yaw)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.timestep)
        
        # --- Close gripper strongly ---
        close_angle = self.calc_gripper_angle(self.gripper_range[0] + 0.001)
        self.move_gripper(self.gripper_range[0] + 0.001, force=2000, max_velocity=0.5)
        for _ in range(200):  # Allow contact to form (longer than TwoFingerGripper for better grip)
            p.stepSimulation()
            time.sleep(self.timestep)
        
        # --- Continuous hold while lifting step-by-step ---
        z_current = grasp_height
        z_target = lift_height
        z_step = (z_target - z_current) / lift_steps
        
        for _ in range(lift_steps):
            z_current += z_step
            self.move(target_pos[0], target_pos[1], z_current, roll, pitch, yaw)
            
            # Continuously reapply strong grip to prevent slip
            p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL,
                                   targetPosition=close_angle, force=2000, maxVelocity=0.5)
            
            p.stepSimulation()
            time.sleep(self.timestep)
    
    def get_position(self):
        pass
    
    def verify_grasp(self, obj):
        pass
        