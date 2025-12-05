import pybullet as p
import pybullet_data
import time
from abc import abstractmethod 
import os

class SceneObject:
    """Base class for any object in the PyBullet scene."""
    def __init__(self, urdf_file, position, orientation=(0, 0, 0)):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = p.getQuaternionFromEuler(orientation)
        self.id = None
        self.name = None

    def load(self):
        """Load the object into the scene."""
        self.id = p.loadURDF(self.urdf_file, self.position, self.orientation)
        return self.id
    
    @abstractmethod
    def update_name(self, name, shape):
        self.name = str(shape) + "_" + str(name)


class Box(SceneObject):
    """Box object inheriting from SceneObject.
    Initialize so we have the instance urdf and position attributes setup with "cube_small.urdf", and a given position.
    You need to reuse parent's initializer accordingly.
    The object should also have a grasp_height attribute set to 0.03 and a name attribute with a string Box_ combined with its id attribute.
    """
    def __init__(self, position, orientation=None):
        # Setup the object
        super().__init__("cube_small.urdf", position)
        self.grasp_height = 0.03
        self.name = f"{self.id}"

    def update_name(self, name, shape="Box"):
        return super().update_name(name, shape)
    

class Cylinder(SceneObject):
    """Cylinder object inheriting from SceneObject.
    Initialize so we have the instance urdf and position attributes setup with "cylinder.urdf", and a given position.
    You need to use parent's initializer accordingly.
    The object should also have a grasp_height attribute set to 0.1 and a name attribute with a string Cylinder_ combined with its id attribute.
    """
    
    def __init__(self, position, orientation=None):
        # Setup the object
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "cylinder.urdf")
        super().__init__(urdf_path, position)
        self.grasp_height = 0.1
        self.name = f"{self.id}"

    def update_name(self, name, shape="cylinder"):
        return super().update_name(name, shape)