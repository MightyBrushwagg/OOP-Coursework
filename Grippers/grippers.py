"""
Gripper module for PyBullet simulation.

This module contains base and specific gripper implementations for robotic grasping
simulations. It includes the base Gripper class and two specific implementations:
TwoFingerGripper and NewGripper (Robotiq 2F-85).
"""

import pybullet as p
import pybullet_data
import time
from abc import ABC, abstractmethod
import numpy as np
import os
from collections import namedtuple
import math

class Gripper(ABC):
    """
    Base gripper class that defines common gripper behavior.
    
    This abstract base class provides common functionality for all gripper types,
    including loading, positioning, and movement operations in PyBullet.
    
    Attributes:
        urdf_path (str): Path to the URDF file defining the gripper model.
        base_position (tuple): Initial position (x, y, z) of the gripper.
        id (int): PyBullet body ID after loading.
        constraint_id (int): PyBullet constraint ID for fixed attachment.
        grasp_moving (bool): Flag indicating if gripper is currently moving.
        orientation (tuple): Initial orientation (roll, pitch, yaw) in radians.
        visuals (str): Visualization mode ("visuals" or "no visuals").
        timestep (float): Simulation timestep (0 for no visuals, 1/240 for visuals).
    """
    def __init__(self, urdf_path, base_position, orientation=(0, 0, 0), visuals="visuals"):
        """
        Initialize the base gripper.
        
        Args:
            urdf_path (str): Path to the URDF file defining the gripper model.
            base_position (tuple): Initial position (x, y, z) of the gripper.
            orientation (tuple, optional): Initial orientation (roll, pitch, yaw) in radians.
                Defaults to (0, 0, 0).
            visuals (str, optional): Visualization mode. "visuals" enables GUI, 
                "no visuals" uses headless mode. Defaults to "visuals".
        """
        self.urdf_path = urdf_path
        self.base_position = base_position
        self.id = None  # Will be set when gripper is loaded
        self.constraint_id = None  # Will be set when gripper is attached
        self.grasp_moving = False
        self.orientation = orientation
        self.visuals = visuals
        # Set timestep based on visualization mode (0 for faster headless simulation)
        if self.visuals == "no visuals":
            self.timestep = 0
        else:
            self.timestep = 1./240.  # 240 Hz simulation rate for smooth visuals

    def load(self):
        """
        Load gripper into the PyBullet world.
        
        Returns:
            int: PyBullet body ID of the loaded gripper.
        """
        self.id = p.loadURDF(self.urdf_path, self.base_position, p.getQuaternionFromEuler(self.orientation))
        return self.id
    
    def attach_fixed(self, offset):
        """
        Attach gripper to a fixed world position using a constraint.
        
        This creates a fixed joint constraint that allows the gripper to be moved
        programmatically while maintaining its attachment to a reference frame.
        
        Args:
            offset (list): Offset position [x, y, z] for the constraint attachment point.
        """
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,  # -1 means base link
            childBodyUniqueId=-1,  # -1 means world frame
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,  # Fixed joint type
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=self.base_position,
            childFrameOrientation=p.getQuaternionFromEuler(self.orientation)
        )

    def move(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Move gripper to a new position and orientation.
        
        Updates the constraint to move the gripper to the specified position
        and orientation. The gripper must be attached first using attach_fixed().
        
        Args:
            x (float): Target x position.
            y (float): Target y position.
            z (float): Target z position.
            roll (float, optional): Roll angle in radians. Defaults to 0.0.
            pitch (float, optional): Pitch angle in radians. Defaults to 0.0.
            yaw (float, optional): Yaw angle in radians. Defaults to 0.0.
            
        Raises:
            ValueError: If gripper has not been attached using attach_fixed().
        """
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")
        p.changeConstraint(
            self.constraint_id,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation=p.getQuaternionFromEuler([roll, pitch, yaw]),
            maxForce=100  # Maximum force to maintain constraint
        )
 
    def update_camera(self, z, yaw):
        """
        Update the debug visualizer camera position and orientation.
        
        Adjusts the camera to follow the gripper's movement, maintaining
        a fixed distance and angle relative to the gripper.
        
        Args:
            z (float): Z coordinate for camera target position.
            yaw (float): Yaw angle in radians to adjust camera angle.
        """
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=50 + (yaw * 180 / 3.1416),  # Convert radians to degrees
            cameraPitch=-60,
            cameraTargetPosition=[0.5, 0.3, z]
        )


class TwoFingerGripper(Gripper):
    """
    A specific two-finger gripper implementation (PR2 gripper).
    
    This class implements a two-finger parallel gripper based on the PR2 robot gripper.
    It provides methods for opening/closing fingers and performing grasp-and-lift operations.
    
    Attributes:
        Inherits all attributes from Gripper class.
    """ 
    def __init__(self, base_position=(0,0,0), orientation=(0, 0, 0), visuals="visuals"):
        """
        Initialize the two-finger gripper.
        
        Args:
            base_position (tuple, optional): Initial position (x, y, z). Defaults to (0,0,0).
            orientation (tuple, optional): Initial orientation (roll, pitch, yaw) in radians.
                Defaults to (0, 0, 0).
            visuals (str, optional): Visualization mode. Defaults to "visuals".
        """
        super().__init__("pr2_gripper.urdf", base_position=base_position, orientation=orientation, visuals=visuals)

    def start(self):
        """
        Initialize gripper to open position at start.
        
        Sets all gripper joints to their initial open positions.
        This should be called after loading the gripper.
        """
        # Predefined joint positions for open gripper state
        initial_positions = [0.550569, 0.0, 0.549657, 0.0]
        for i, pos in enumerate(initial_positions):
            p.resetJointState(self.id, i, pos)

    def open(self):
        """
        Open the gripper fingers.
        
        Commands the finger joints (0 and 2) to move to their open position (0.0).
        """
        # Joints 0 and 2 are the finger joints
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, maxVelocity=1, force=10)

    def close(self):
        """
        Close the gripper fingers.
        
        Commands the finger joints to move to their closed position (0.1 radians).
        """
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.1, maxVelocity=1, force=10)
    
    def generate_angles(self, gripper_pos, object_pos):
        """
        Calculate gripper orientation angles to point towards an object.
        
        Computes the roll, pitch, and yaw angles needed to orient the gripper
        such that it points from gripper_pos towards object_pos.
        
        Args:
            gripper_pos (list): Gripper position [x, y, z].
            object_pos (list): Target object position [x, y, z].
            
        Returns:
            numpy.ndarray: Array of [roll, pitch, yaw] angles in radians.
        """
        # Calculate direction vector from gripper to object
        direction = np.array(object_pos) - np.array(gripper_pos)
        direction = direction/np.linalg.norm(direction)  # Normalize

        # Calculate Euler angles from direction vector
        pitch = np.arcsin(-direction[2])  # Vertical angle
        yaw = np.arctan2(direction[1], direction[0])  # Horizontal angle
        roll = np.pi  # Fixed roll angle for this gripper type

        return np.array([roll, pitch, yaw])

    def target_position(self, obj):
        """
        Calculate target position and approach position for grasping.
        
        Computes the optimal target position (with offset) and approach position
        for grasping an object. The offset ensures the gripper approaches from
        the correct angle relative to the object.
        
        Args:
            obj: Object instance with a position attribute.
            
        Returns:
            tuple: (target_pos, approach_pos) where:
                - target_pos: numpy array [x, y, z] - final grasp position with offset
                - approach_pos: list [x, y, z] - initial approach position above object
        """
        # Calculate approach position (slightly above gripper base)
        approach_pos = [self.base_position[0], self.base_position[1], self.base_position[2] + 0.1]
        off = 0.04  # Offset distance for approach angle (tunable parameter)
        
        # Calculate direction vector from object to approach position (horizontal only)
        direction = np.array(approach_pos) - np.array(obj.position)
        direction[2] = 0  # No z term offset - only horizontal offset needed

        # Normalize direction and apply offset
        if np.linalg.norm(direction) > 0:  # Avoid division by zero
            direction = direction / np.linalg.norm(direction)  # Normalize direction vector
            offset = direction * off  # Scale by offset distance
        else:
            offset = np.array([0, 0, 0])  # No offset if positions are identical

        target_pos = np.array(obj.position) + offset  # Apply offset to object position
        return target_pos, approach_pos

    def grasp_and_lift(self, obj, lift_height=0.4, lift_steps=150):
        """
        Perform a complete grasping and lifting sequence with sustained gripping force.
        
        Executes a multi-phase grasp operation:
        1. Move above object
        2. Lower to grasp height
        3. Close gripper with strong force
        4. Lift object while maintaining grip
        
        Args:
            obj: Object instance to grasp (must have position and grasp_height attributes).
            lift_height (float, optional): Height to lift object to. Defaults to 0.4.
            lift_steps (int, optional): Number of simulation steps for lifting. Defaults to 150.
        """
        # Calculate target position and approach angles
        target_pos, approach_pos = self.target_position(obj)
        roll, pitch, yaw = self.generate_angles(approach_pos, target_pos)
        grasp_height = obj.grasp_height

        # --- Set friction on contact surfaces ---
        # High friction helps maintain grip during lifting
        p.changeDynamics(obj.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)

        # --- Phase 1: Move above object ---
        self.move(target_pos[0], target_pos[1], target_pos[2] + 0.2, roll, pitch, yaw)
        for _ in range(100):  # Allow time for movement
            p.stepSimulation()
            time.sleep(self.timestep)

        # --- Phase 2: Lower onto object ---
        self.move(target_pos[0], target_pos[1], grasp_height, roll, pitch, yaw)
        for _ in range(100):  # Allow time for descent
            p.stepSimulation()
            time.sleep(self.timestep)

        # --- Phase 3: Close gripper strongly ---
        # Use high force to ensure secure grip
        for joint in [0, 2]:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.12, force=300, maxVelocity=2)
        for _ in range(100):  # Allow contact to form between gripper and object
            p.stepSimulation()
            time.sleep(self.timestep)

        # --- Phase 4: Continuous hold while lifting step-by-step ---
        z_current = grasp_height
        z_target = lift_height
        z_step = (z_target - z_current) / lift_steps  # Calculate incremental step size

        for _ in range(lift_steps):
            z_current += z_step
            self.move(target_pos[0], target_pos[1], z_current, roll, pitch, yaw)

            # Continuously reapply strong grip to prevent slip during lifting
            for joint in [0, 2]:
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                        targetPosition=0.12, force=400, maxVelocity=2)

            p.stepSimulation()
            time.sleep(self.timestep)

    def get_position(self):
        """
        Get current gripper position.
        
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass

    def verify_grasp(self, obj):
        """
        Verify if object is successfully grasped.
        
        Args:
            obj: Object instance to check.
            
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass

class NewGripper(Gripper):
    """
    Robotiq 2F-85 gripper implementation.
    
    This class implements a more sophisticated two-finger gripper with mimic joints.
    The gripper uses a parent joint with multiple child joints that mimic its motion
    through gear constraints.
    
    Attributes:
        mimic_parent_id (int): Joint ID of the parent joint controlling gripper opening.
        mimic_child_multiplier (dict): Dictionary mapping child joint IDs to their
            gear ratio multipliers.
        gripper_range (list): [min_open, max_open] in meters. Defaults to [0, 0.085].
    """
    def __init__(self, base_position=(0,0,0), orientation=(0, 0, 0), visuals="visuals"):
        """
        Initialize the Robotiq 2F-85 gripper.
        
        Args:
            base_position (tuple, optional): Initial position (x, y, z). Defaults to (0,0,0).
            orientation (tuple, optional): Initial orientation (roll, pitch, yaw) in radians.
                Defaults to (0, 0, 0).
            visuals (str, optional): Visualization mode. Defaults to "visuals".
        """
        # Get the correct URDF path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "NewGripper", "robotiq_2f_85", "robotiq.urdf")
        super().__init__(urdf_path, base_position=base_position, orientation=orientation, visuals=visuals)
        self.mimic_parent_id = None  # Will be set in start() method
        self.mimic_child_multiplier = {}  # Will be populated in start() method
        self.gripper_range = [0, 0.085]  # Gripper opening range in meters [min, max]

    def start(self):
        """
        Initialize gripper joints and setup mimic joint constraints.
        
        This method:
        1. Parses all joints from the URDF
        2. Identifies the parent joint and child joints
        3. Creates gear constraints to link child joints to parent
        4. Opens the gripper to initial position
        
        Must be called after loading the gripper.
        """
        # Parse all joints from the URDF model
        num_joints = p.getNumJoints(self.id)
        JointInfo = namedtuple('JointInfo', ['id', 'name', 'type', 'lower', 'upper', 'maxForce'])
        joints = []
        
        for i in range(num_joints):
            info = p.getJointInfo(self.id, i)
            jid = info[0]
            name = info[1].decode()  # Decode bytes to string
            jtype = info[2]
            lower = info[8]  # Lower joint limit
            upper = info[9]  # Upper joint limit
            maxForce = info[10]  # Maximum joint force
            joints.append(JointInfo(jid, name, jtype, lower, upper, maxForce))
            # Disable default motor control to allow manual control
            p.setJointMotorControl2(self.id, jid, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        
        # Setup mimic joints - define parent and children with gear ratios
        mimic_parent_name = 'finger_joint'  # Main control joint
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,   # Positive gear ratio
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,    # Negative gear ratio (opposite motion)
            'right_inner_finger_joint': -1
        }
        
        # Find parent joint ID
        self.mimic_parent_id = [j.id for j in joints if j.name == mimic_parent_name][0]
        # Create mapping of child joint IDs to their multipliers
        self.mimic_child_multiplier = {j.id: mimic_children_names[j.name] 
                                       for j in joints if j.name in mimic_children_names}
        
        # Create gear constraints to link child joints to parent
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id, self.mimic_parent_id,  # Parent body and joint
                self.id, joint_id,              # Child body and joint
                jointType=p.JOINT_GEAR,         # Gear joint type for mimic motion
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            # Set gear ratio (negative for opposite motion)
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)
        
        # Initialize gripper to open position
        self.open()
    
    def calc_gripper_angle(self, open_length):
        """
        Calculate gripper joint angle from opening length.
        
        Converts a desired opening distance (in meters) to the corresponding
        joint angle using the kinematic model of the Robotiq gripper.
        
        Args:
            open_length (float): Desired opening distance in meters.
            
        Returns:
            float: Joint angle in radians.
        """
        # Kinematic formula for Robotiq 2F-85 gripper
        return 0.715 - math.asin((open_length - 0.010) / 0.1143)
    
    def move_gripper(self, open_length, force=100, max_velocity=None):
        """
        Move gripper to specified opening length.
        
        Commands the parent joint to move to the angle corresponding to the
        desired opening length. Child joints will follow via mimic constraints.
        
        Args:
            open_length (float): Desired opening distance in meters.
            force (float, optional): Maximum force to apply. Defaults to 100.
            max_velocity (float, optional): Maximum joint velocity. If None, no limit.
        """
        open_angle = self.calc_gripper_angle(open_length)
        if max_velocity is not None:
            p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL,
                                    targetPosition=open_angle, force=force, maxVelocity=max_velocity)
        else:
            p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL,
                                    targetPosition=open_angle, force=force)
    
    def open(self):
        """
        Open the gripper fully.
        
        Moves gripper to maximum opening distance.
        """
        self.move_gripper(self.gripper_range[1])
    
    def close(self, force=2000, max_velocity=0.5):
        """
        Close the gripper with specified force.
        
        Moves gripper to minimum opening (with small gap to prevent collision).
        Uses high force to ensure secure grip.
        
        Args:
            force (float, optional): Maximum closing force. Defaults to 2000.
            max_velocity (float, optional): Maximum closing velocity. Defaults to 0.5.
        """
        target_close_position = self.gripper_range[0] + 0.001  # Leave 1mm gap to prevent collision
        self.move_gripper(target_close_position, force=force, max_velocity=max_velocity)

    def generate_angles(self, gripper_pos, object_pos):
        """
        Calculate gripper orientation angles to point towards an object.
        
        Similar to TwoFingerGripper but with inverted pitch for this gripper's
        coordinate frame.
        
        Args:
            gripper_pos (list): Gripper position [x, y, z].
            object_pos (list): Target object position [x, y, z].
            
        Returns:
            numpy.ndarray: Array of [roll, pitch, yaw] angles in radians.
        """
        # Calculate direction vector from gripper to object
        direction = np.array(object_pos) - np.array(gripper_pos)
        direction = direction/np.linalg.norm(direction)  # Normalize

        # Calculate Euler angles (pitch is inverted for this gripper)
        pitch = np.arcsin(-direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        roll = np.pi

        return np.array([roll, -pitch, yaw])  # Note: pitch is negated
    
    def target_position(self, obj):
        """
        Calculate target position and approach position for grasping.
        
        Similar to TwoFingerGripper but with different offset parameters
        optimized for the Robotiq gripper geometry.
        
        Args:
            obj: Object instance with a position attribute.
            
        Returns:
            tuple: (target_pos, approach_pos) where:
                - target_pos: numpy array [x, y, z] - final grasp position with offset
                - approach_pos: list [x, y, z] - initial approach position
        """
        # Calculate approach position (offset from base)
        approach_pos = [self.base_position[0] - 0.2, self.base_position[1], self.base_position[2] + 0.1]
        off = -0.12  # Offset distance (negative for different approach direction)
        
        # Calculate horizontal direction vector
        direction = np.array(approach_pos) - np.array(obj.position)
        direction[2] = 0  # Only horizontal offset

        # Normalize and apply offset
        if np.linalg.norm(direction) > 0:  # Avoid division by zero
            direction = direction / np.linalg.norm(direction)
            offset = direction * off
        else:
            offset = np.array([0, 0, 0])

        target_pos = np.array(obj.position) + offset
        return target_pos, approach_pos
    
    def grasp_and_lift(self, obj, lift_height=0.4, lift_steps=150):
        """
        Perform a complete grasping and lifting sequence with sustained gripping force.
        
        Executes a multi-phase grasp operation optimized for the Robotiq gripper:
        1. Move above object
        2. Lower to grasp height
        3. Close gripper with strong force
        4. Lift object while maintaining grip
        
        Uses higher friction values and longer contact formation time than TwoFingerGripper
        for more reliable grasping.
        
        Args:
            obj: Object instance to grasp (must have position and grasp_height attributes).
            lift_height (float, optional): Height to lift object to. Defaults to 0.4.
            lift_steps (int, optional): Number of simulation steps for lifting. Defaults to 150.
        """
        # Calculate target position and approach angles
        target_pos, approach_pos = self.target_position(obj)
        roll, pitch, yaw = self.generate_angles(approach_pos, target_pos)
       
        # Adjust grasp height for this gripper's geometry
        grasp_height = obj.grasp_height + 0.2
        
        # --- Set friction on contact surfaces ---
        # Higher friction values work better for Robotiq gripper
        friction_params = {
            'lateralFriction': 20.0,      # High lateral friction
            'rollingFriction': 0.5,
            'spinningFriction': 0.5,
            'contactStiffness': 10000,    # Stiff contact for better grip
            'contactDamping': 2000
        }
        # Apply friction to all links of object and gripper
        for i in range(-1, p.getNumJoints(obj.id)):  # -1 is base link
            p.changeDynamics(obj.id, i, **friction_params)
        for i in range(-1, p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, **friction_params)
        
        # --- Phase 1: Move above object ---
        self.move(target_pos[0], target_pos[1], target_pos[2] + 0.3, roll, pitch, yaw)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.timestep)
        
        # --- Phase 2: Lower onto object ---
        self.move(target_pos[0], target_pos[1], grasp_height, roll, pitch, yaw)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.timestep)
        
        # --- Phase 3: Close gripper strongly ---
        close_angle = self.calc_gripper_angle(self.gripper_range[0] + 0.001)
        self.move_gripper(self.gripper_range[0] + 0.001, force=2000, max_velocity=0.5)
        # Longer contact formation time for better grip
        for _ in range(200):
            p.stepSimulation()
            time.sleep(self.timestep)
        
        # --- Phase 4: Continuous hold while lifting step-by-step ---
        z_current = grasp_height
        z_target = lift_height
        z_step = (z_target - z_current) / lift_steps
        
        for _ in range(lift_steps):
            z_current += z_step
            self.move(target_pos[0], target_pos[1], z_current, roll, pitch, yaw)
            
            # Continuously reapply strong grip to prevent slip during lifting
            p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL,
                                   targetPosition=close_angle, force=2000, maxVelocity=0.5)
            
            p.stepSimulation()
            time.sleep(self.timestep)
    
    def get_position(self):
        """
        Get current gripper position.
        
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass
    
    def verify_grasp(self, obj):
        """
        Verify if object is successfully grasped.
        
        Args:
            obj: Object instance to check.
            
        Returns:
            NotImplemented: Method to be implemented in future versions.
        """
        pass
        