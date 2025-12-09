"""
Simulation module for PyBullet robotic grasping simulations.

This module provides the Simulation class which manages the PyBullet physics
environment, coordinates gripper-object interactions, and collects success/failure
data for machine learning training.
"""

import pybullet as p
import pybullet_data
import time
from abc import abstractmethod
import pandas as pd
import numpy as np
from Objects.objects import Box, Cylinder
from Grippers.grippers import TwoFingerGripper, NewGripper
import random
from Data.data import Data

class Simulation:
    """
    Main simulation class for robotic grasping experiments.
    
    Manages the PyBullet physics simulation environment, creates scenes with
    grippers and objects, executes grasp attempts, and collects success/failure
    data. The simulation can run in visual or headless mode.
    
    Attributes:
        data (Data): Data object for storing simulation results.
        positionSuccess (dict): Dictionary mapping positions to success status.
        visuals (str): Visualization mode ("visuals" or "no visuals").
        object (str): Type of object being used ("cube" or "cylinder").
        gripper (str): Type of gripper being used ("two_finger" or "new_gripper").
        timestep (float): Simulation timestep (0 for headless, 1/240 for visual).
        file_save (str): Filename for saving simulation data.
        step_threshold (int): Number of simulation steps required for successful grasp.
        step_count (int): Current count of successful grasp steps.
        iterations (int): Total number of simulation iterations to run.
        plane_id (int): PyBullet body ID of the ground plane.
    """

    # Mapping of object type strings to class constructors
    obj_dic = {
        "cube": Box,
        "cylinder": Cylinder
    }

    # Mapping of gripper type strings to class constructors
    gripper_dic = {
        "two_finger": TwoFingerGripper,
        "new_gripper": NewGripper
    }

    def __init__(self, iterations, object="cube", gripper="two_finger", visuals = "visuals", file_save=None, data=None):
        """
        Initialize the simulation environment.
        
        Args:
            iterations (int): Number of simulation iterations to run.
            object (str, optional): Object type ("cube" or "cylinder"). Defaults to "cube".
            gripper (str, optional): Gripper type ("two_finger" or "new_gripper").
                Defaults to "two_finger".
            visuals (str, optional): Visualization mode. Defaults to "visuals".
            file_save (str, optional): Filename for saving data. Auto-generated if None.
            data (Data, optional): Existing Data object to use. Creates new one if None.
        """
        # Initialize data collection (allocate extra space for filtering)
        self.data = Data(num_points=iterations*2 + 100) if data is None else data
        self.positionSuccess = {}  # Legacy: track position-based success
        self.visuals = visuals
        self.object = object
        self.gripper = gripper
        
        # Set simulation timestep based on visualization mode
        if visuals == "no visuals":
            self.timestep = 0  # No delay for faster headless simulation
        else:
            self.timestep = 1./240.  # 240 Hz for smooth visual simulation
            
        # Set output filename
        self.file_save = file_save if file_save is not None else f"{object}-{gripper}-data.csv"
        
        # Initialize PyBullet simulation environment
        self.start_simulation()
        
        # Grasp verification parameters
        self.step_threshold = 3*240  # 3 seconds at 240Hz = 720 steps for successful grasp
        self.step_count = 0
        self.iterations = iterations
        
        # self.run_simulations(self.iterations, object, gripper, self.step_threshold, self.step_count)
        # p.disconnect()
        # print(self.data.data)
        # self.save_data()
        # self.data.statistics()
        # self.data.visualise_data()
        
    def run_simulations(self, iterations=None, object=None, gripper=None, step_threshold=None, step_count=None, save=True):
        """
        Run multiple simulation iterations to collect grasping data.
        
        For each iteration:
        1. Creates a scene with object and gripper at specified position/orientation
        2. Executes grasp-and-lift sequence
        3. Monitors contact points to verify successful grasp
        4. Records success/failure in data object
        5. Resets scene for next iteration
        
        Success criteria:
        - At least 2 contact points between gripper and object
        - No contact points between object and ground plane
        - Maintained for at least step_threshold simulation steps
        
        Args:
            iterations (int, optional): Number of iterations. Uses self.iterations if None.
            object (str, optional): Object type. Uses self.object if None.
            gripper (str, optional): Gripper type. Uses self.gripper if None.
            step_threshold (int, optional): Steps required for success. Uses self.step_threshold if None.
            step_count (int, optional): Initial step count. Uses self.step_count if None.
            save (bool, optional): Whether to save data and visualization. Defaults to True.
        """
        # Use instance defaults if parameters not provided
        iterations = self.iterations if iterations is None else iterations
        object = self.object if object is None else object
        gripper = self.gripper if gripper is None else gripper  
        step_threshold = self.step_threshold if step_threshold is None else step_threshold
        step_count = self.step_count if step_count is None else step_count
        
        # Run simulation iterations
        for i in range(iterations):
            # Get position and orientation from pre-generated data
            run_data = self.data.data.loc[i]
            pos = [run_data["x"], run_data["y"], run_data["z"]]
            orientations = [run_data["roll"], run_data["pitch"], run_data["yaw"]]
            
            verify_once = True  # Flag to ensure we only record success once per iteration
            # Create scene and execute grasp attempt
            self.run_one(object, gripper, gripper_pos=pos, gripper_ori=orientations)
            
            # Monitor simulation for successful grasp
            for j in range(2000):  # Maximum simulation steps per iteration
                p.stepSimulation()
                time.sleep(self.timestep)
                
                # Check contact points: gripper-object and object-ground
                contact_points_gripper = p.getContactPoints(self.gripper.id, self.object.id)
                contact_points_plane = p.getContactPoints(self.object.id, self.plane_id)
                
                # Success condition: gripper has object (>=2 contacts) and object is lifted (no ground contact)
                if len(contact_points_gripper) >= 2 and len(contact_points_plane) == 0:
                    step_count += 1
                    # Verify grasp is maintained for threshold duration
                    if step_count >= step_threshold and verify_once == True:
                        verify_once = False  # Mark as successful
                        break
                # else:
                #     step_count = 0  # Reset counter if condition not met
            
            # Clean up scene for next iteration
            self.reset_scene()
            
            # Record success/failure in data
            self.data.update_success(i, success=(not verify_once))
        
        # Save results if requested
        self.data.statistics()
        if save:
            self.save_data("Data/" + self.file_save)
            # Generate visualization plot
            # save the plot in the Data folder with the same name as the csv but with .jpg extension
            self.data.visualise_data("Data/" + self.file_save.replace(".csv", ".jpg"), title=f"Gripper Positions for {object} with {gripper}")
            # self.data.visualise_data("Data/" + self.file_save.replace(".csv", ".jpg")) 

        else:
            self.data.visualise_data(file_name=None, title=f"Gripper Positions for {object} with {gripper}")

    def start_simulation(self):
        """
        Initialize the PyBullet physics simulation environment.
        
        Sets up the physics engine, loads the ground plane, and configures
        simulation parameters. Must be called before running simulations.
        
        Returns:
            int: PyBullet body ID of the ground plane.
        """
        # Connect to PyBullet (GUI for visuals, DIRECT for headless)
        visual_dict = {"visuals": p.GUI, "no visuals": p.DIRECT}
        p.connect(visual_dict[self.visuals])
        
        # Add PyBullet data path for loading URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Reset simulation to clean state
        p.resetSimulation()
        
        # Set gravity (negative z direction)
        p.setGravity(0, 0, -10)
        
        # Disable real-time simulation (step manually for control)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        return self.plane_id
    
    def run_one(self, object, gripper, gripper_pos = [0,0,0], gripper_ori=[0,0,0]):
        """
        Run a single simulation iteration.
        
        Creates a scene and executes one grasp attempt with the specified
        gripper position and orientation.
        
        Args:
            object (str): Object type to use.
            gripper (str): Gripper type to use.
            gripper_pos (list, optional): Gripper position [x, y, z]. Defaults to [0,0,0].
            gripper_ori (list, optional): Gripper orientation [roll, pitch, yaw].
                Defaults to [0,0,0].
        """
        self.create_scene(object=object, gripper=gripper, gripper_pos=gripper_pos, gripper_ori=gripper_ori)
        
    def save_data(self, name=None):
        """
        Save simulation data to CSV file.
        
        Args:
            name (str, optional): Filename to save to. Uses self.file_save if None.
        """
        if name is None:
            self.data.upload_data(self.file_save)
        else:
            self.data.upload_data(name)

    def create_scene(self, object, gripper, gripper_pos = [0,0,0], gripper_ori=[0,0,0]):
        """
        Create a simulation scene with object and gripper.
        
        Instantiates the specified object and gripper, loads them into PyBullet,
        attaches the gripper, and executes the grasp-and-lift sequence.
        
        Args:
            object (str): Object type ("cube" or "cylinder").
            gripper (str): Gripper type ("two_finger" or "new_gripper").
            gripper_pos (list, optional): Gripper base position [x, y, z]. Defaults to [0,0,0].
            gripper_ori (list, optional): Gripper orientation [roll, pitch, yaw]. Defaults to [0,0,0].
        """
        # Create object at fixed position (centered on ground)
        self.object = Simulation.obj_dic[object]([0, 0, 0.075])
        # Create gripper at specified position and orientation
        self.gripper = Simulation.gripper_dic[gripper](
            base_position=gripper_pos, 
            orientation=gripper_ori, 
            visuals=self.visuals
        )
        
        # Load object into simulation
        obj_id = self.object.load()
        self.object.update_name(obj_id)

        # Load and initialize gripper
        self.gripper.load()
        self.gripper.start()  # Set to initial open position
        
        # Attach gripper to fixed constraint (allows programmatic movement)
        self.gripper.attach_fixed(offset=[0.2, 0, 0])
        
        # Execute grasp-and-lift sequence
        self.gripper.grasp_and_lift(self.object)


    def reset_scene(self):
        """
        Clean up the current simulation scene.
        
        Removes the gripper constraint, gripper body, and object body
        from the simulation to prepare for the next iteration.
        """
        # Remove gripper constraint if it exists
        if self.gripper.constraint_id is not None:
            p.removeConstraint(self.gripper.constraint_id)
        
        # Remove bodies from simulation
        p.removeBody(self.gripper.id)
        p.removeBody(self.object.id)