import pybullet as p
import pybullet_data
import time
from abc import abstractmethod
import pandas as pd
import numpy as np
from objects import Box, Cylinder
from grippers import TwoFingerGripper
from RandDataset import pos
import random

class Simulation:

    obj_dic = {
        "box": Box,
        "cylinder": Cylinder
    }

    gripper_dic = {
        "two_finger": TwoFingerGripper
    }

    def __init__(self, iterations, data=None, object="box", gripper="two_finger"):

        self.start_simulation()

        for i in range(iterations):
            self.run_one(object, gripper)
            for j in range(1000):
                p.stepSimulation()
                time.sleep(1./240.)
            self.reset_scene()

        p.disconnect()

    def start_simulation(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
        plane_id = p.loadURDF("plane.urdf")
        return plane_id
    
    def run_one(self, object, gripper):
        self.create_scene(object=object, gripper=gripper)
        

    def save_data(self):
        pass

    def upload_data(self):
        pass

    def create_scene(self, object, gripper):
        self.object = Simulation.obj_dic[object]([0,0,0])
        self.gripper = Simulation.gripper_dic[gripper](pos[random.randint(0, len(pos) - 1)])
        obj_id = self.object.load()
        self.object.update_name(obj_id)

        
        self.gripper.load()
        self.gripper.start()
        self.gripper.attach_fixed(offset=[0.2, 0, 0])
        # self.gripper.update_camera(z=0.7, yaw=0.0) # Allows camera free roam
        self.gripper.grasp_and_lift(self.object)


    def reset_scene(self):
        if self.gripper.constraint_id is not None:
            p.removeConstraint(self.gripper.constraint_id)
        p.removeBody(self.gripper.id)

        p.removeBody(self.object.id)

    def set_random_position(self):
        # input: current dataset?

        # return random position and orientation
        pass