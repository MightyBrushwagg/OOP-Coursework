"""
Object module for PyBullet simulation.

This module provides base and specific object classes for use in robotic
grasping simulations. Objects can be loaded into PyBullet and have properties
like grasp_height that define how they should be grasped.
"""

import pybullet as p
import pybullet_data
import time
from abc import abstractmethod
import os

class SceneObject:
    """
    Base class for any object in the PyBullet scene.
    
    Provides common functionality for loading and managing objects in the
    simulation environment. Subclasses define specific object types (Box, Cylinder).
    
    Attributes:
        urdf_file (str): Path to URDF file defining the object model.
        position (list): Object position [x, y, z] in world coordinates.
        orientation (tuple): Object orientation as quaternion.
        id (int): PyBullet body ID after loading.
        name (str): Object name identifier.
    """
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        """
        Initialize the scene object.
        
        Args:
            urdf_file (str): Path to URDF file defining the object model.
            position (list): Initial position [x, y, z] in world coordinates.
            orientation (tuple, optional): Initial orientation (roll, pitch, yaw) in radians.
                Defaults to (0, 0, 0).
        """
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)  # Convert to quaternion
        self.id = None  # Will be set when object is loaded
        self.name = None  # Will be set by update_name()

    def load(self):
        """
        Load the object into the PyBullet scene.
        
        Returns:
            int: PyBullet body ID of the loaded object.
        """
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id
    
    @abstractmethod
    def update_name(self, name, shape):
        """
        Update the object's name identifier.
        
        Args:
            name: Object identifier (typically the PyBullet body ID).
            shape: Object shape type (e.g., "Box", "Cylinder").
        """
        self.name = str(shape) + "_" + str(name)


class Box(SceneObject):
    """
    Box (cube) object for simulation.
    
    Represents a cubic object that can be grasped. Uses the default PyBullet
    cube_small.urdf model. The grasp_height defines the optimal z-coordinate
    for grasping this object.
    
    Attributes:
        grasp_height (float): Optimal z-coordinate for grasping (0.03m for cube).
        name (str): Object name identifier.
    """
    def __init__(self, position, orientation=None):
        """
        Initialize the box object.
        
        Args:
            position (list): Initial position [x, y, z] in world coordinates.
            orientation (tuple, optional): Initial orientation. Defaults to None.
        """
        super().__init__("cube_small.urdf", position)
        self.grasp_height = 0.03  # Optimal grasp height for cube (3cm from bottom)
        self.name = f"{self.id}"  # Temporary name, will be updated after loading

    def update_name(self, name, shape="Box"):
        """
        Update the object name with shape prefix.
        
        Args:
            name: Object identifier (typically PyBullet body ID).
            shape (str, optional): Shape type. Defaults to "Box".
            
        Returns:
            str: Updated name string.
        """
        return super().update_name(name, shape)
    

class Cylinder(SceneObject):
    """
    Cylinder object for simulation.
    
    Represents a cylindrical object that can be grasped. Uses a custom
    cylinder.urdf model. The grasp_height defines the optimal z-coordinate
    for grasping this object.
    
    Attributes:
        grasp_height (float): Optimal z-coordinate for grasping (0.1m for cylinder).
        name (str): Object name identifier.
    """
    
    def __init__(self, position, orientation=None):
        """
        Initialise the cylinder object.
        
        Args:
            position (list): Initial position [x, y, z] in world coordinates.
            orientation (tuple, optional): Initial orientation. Defaults to None.
        """
        # Get path to cylinder URDF file (located in same directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "cylinder.urdf")
        super().__init__(urdf_path, position)
        self.grasp_height = 0.1  # Optimal grasp height for cylinder (10cm from bottom)
        self.name = f"{self.id}"  # Temporary name, will be updated after loading

    def update_name(self, name, shape="cylinder"):
        """
        Update the object name with shape prefix.
        
        Args:
            name: Object identifier (typically PyBullet body ID).
            shape (str, optional): Shape type. Defaults to "cylinder".
            
        Returns:
            str: Updated name string.
        """
        return super().update_name(name, shape)