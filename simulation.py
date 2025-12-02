import pybullet as p
import pybullet_data
import time
from abc import abstractmethod
import pandas as pd
import numpy as np
from objects import Box, Cylinder
from grippers import TwoFingerGripper, NewGripper
import random
from data import Data

class Simulation:

    obj_dic = {
        "cube": Box,
        "cylinder": Cylinder
    }

    gripper_dic = {
        "two_finger": TwoFingerGripper,
        "new_gripper": NewGripper
    }

    def __init__(self, iterations, object="cube", gripper="two_finger", visuals = "visuals"):
        self.data = Data(num_points=iterations*2 + 100)
        self.positionSuccess = {}
        self.visuals = visuals
        if visuals == "no visuals":
            self.timestep = 0
        else:
            self.timestep = 1./240.
        self.start_simulation()
        step_threshold = 3*240 # 3 seconds at 240Hz = 720
        step_count = 0
        self.iterations = iterations
        self.run_simulations(self.iterations, object, gripper, step_threshold, step_count)
        p.disconnect()
        print(self.data.data)
        self.save_data()
        self.data.statistics()
        self.data.visualise_data()

    def run_simulations(self, iterations, object, gripper, step_threshold, step_count):
        for i in range(iterations):
            run_data = self.data.data.loc[i]
            # print(f"run data: {run_data}")
            pos = [run_data["x"], run_data["y"], run_data["z"]]
            orientations = [run_data["roll"], run_data["pitch"], run_data["yaw"]]
            # print(f"pos: {pos}")
            verify_once = True
            self.run_one(object, gripper, gripper_pos=pos, gripper_ori=orientations)
            for j in range(2000):
                p.stepSimulation()
                time.sleep(self.timestep)
                contact_points_gripper = p.getContactPoints(self.gripper.id, self.object.id)
                contact_points_plane = p.getContactPoints(self.object.id, self.plane_id)
                if len(contact_points_gripper) >= 2 and len(contact_points_plane) == 0:
                    step_count += 1
                    if step_count >= step_threshold and verify_once == True:
                        print("Success")
                        # self.positionSuccess[tuple(self.startPosition)] = True
                        verify_once = False
                        break
            self.reset_scene()
            if verify_once == True:
                print("Failure")
                # self.positionSuccess[tuple(self.startPosition)] = False
            else:
                print("Success")
            self.data.update_success(i, success=(not verify_once))
        print(self.positionSuccess.values())
# Add in ground contacts + limit gripper contact to 2 

    def start_simulation(self):
        visual_dict={"visuals": p.GUI, "no visuals": p.DIRECT}
        p.connect(visual_dict[self.visuals]) # GUI = visual, Direct = no visuals = Faster
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
        self.plane_id = p.loadURDF("plane.urdf")
        return self.plane_id
    
    def run_one(self, object, gripper, gripper_pos = [0,0,0], gripper_ori=[0,0,0]):
        self.create_scene(object=object, gripper=gripper, gripper_pos=gripper_pos, gripper_ori=gripper_ori)
        
    def save_data(self, name = "cube-twofingergripper"):
        self.data.upload_data(name)

    def create_scene(self, object, gripper, gripper_pos = [0,0,0], gripper_ori=[0,0,0]):
        self.object = Simulation.obj_dic[object]([0,0,0.075])
        self.gripper = Simulation.gripper_dic[gripper](base_position=gripper_pos, orientation=gripper_ori, visuals=self.visuals) # , orientation=gripper_ori
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