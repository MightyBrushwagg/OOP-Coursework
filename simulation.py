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
        self.positionSuccess = {}
        self.start_simulation()
        step_threshold = 1200 # 5 seconds at 240Hz
        step_count = 0
        for i in range(iterations):
            self.startPosition = pos[random.randint(0, len(pos) - 1)]
            verify_once = True
            self.run_one(object, gripper)
            for j in range(2000):
                p.stepSimulation()
                #time.sleep(1./240.)
                contact_points = p.getContactPoints(self.gripper.id, self.object.id)
                if len(contact_points) > 0:
                    step_count += 1
                    if step_count >= step_threshold and verify_once == True:
                        print("Grasped Successfully")
                        self.positionSuccess[tuple(self.startPosition)] = True
                        verify_once = False
            self.reset_scene()
        print(self.positionSuccess)
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
        self.gripper = Simulation.gripper_dic[gripper](self.startPosition)
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