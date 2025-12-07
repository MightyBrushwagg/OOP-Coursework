"""
Data management module for simulation results.

This module provides the Data class for generating, storing, processing, and
visualizing grasping simulation data. It handles data generation, import/export,
statistical analysis, and 3D visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Data:
    """
    Data management class for simulation results.
    
    Handles generation of random gripper positions/orientations, storage of
    simulation results, data splitting for ML training, and visualization.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing columns:
            - x, y, z: Gripper position coordinates
            - roll, pitch, yaw: Gripper orientation angles
            - success: Boolean indicating grasp success/failure
    """
    def __init__(self, data=None, num_points=500, object_pos=[0,0,0]):
        if data == None:
            column_tmp  = {"x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": [], "success": []}
            self.data = pd.DataFrame(column_tmp)
            all_points, all_angles = self.make_data(object_pos=object_pos, num_points=num_points)
            # print(all_angles.shape)
            self.data["x"] = all_points[:,0]
            self.data["y"] = all_points[:,1]
            self.data["z"] = all_points[:,2]
            self.data["roll"] = all_angles[:,0]
            self.data["pitch"] = all_angles[:,1]
            self.data["yaw"] = all_angles[:,2]
            # print(self.data)
        else:
            self.data = data

    def make_data(self, object_pos, num_points=400, R=0.5, height=0):
        """
        Generate random gripper positions and orientations.
        
        Creates a uniform distribution of points on a sphere (with noise) around
        the object position, then calculates appropriate gripper orientations
        to point towards the object.
        
        Args:
            object_pos (list): Object position [x, y, z].
            num_points (int, optional): Number of points to generate. Defaults to 400.
            R (float, optional): Radius of sphere for point distribution. Defaults to 0.5.
            height (float, optional): Minimum z-coordinate (filters out points below).
                Defaults to 0.
            
        Returns:
            tuple: (valid_points, orientations) where:
                - valid_points: numpy array of shape (N, 3) with valid positions
                - orientations: numpy array of shape (N, 3) with [roll, pitch, yaw] angles
        """
        # Step 1: Generate random 3D Gaussian vectors
        coords = np.random.normal(size=(3, num_points))
        
        # Step 2: Calculate the magnitude of each vector
        distance_from_origin = np.linalg.norm(coords, ord=2, axis=0)
        
        # Step 3: Normalize the vectors to get uniform distribution on unit sphere
        normalised_coords = coords / distance_from_origin
        
        # Step 4: Scale to desired radius
        points = R * normalised_coords
        
        # Step 5: Add Gaussian noise for more realistic distribution
        noise = np.random.normal(0, 0.02, size=(3, num_points))
        noisy = points + noise
        
        # Step 6: Filter points above minimum height (remove points below ground)
        valid = np.array([[coord[0], coord[1], coord[2]] 
                         for coord in noisy.T if coord[2] >= height])
        
        # Step 7: Calculate orientations for each valid position
        orientations = np.array([self.generate_angle(position, object_pos) 
                                for position in valid])
        
        return valid, orientations
    
    def import_data(self, path):
        """
        Import data from CSV file.
        
        Args:
            path (str): Path to CSV file containing simulation data.
        """
        self.data = pd.read_csv(path)
        # print("Columns in imported data:", self.data.columns.tolist())  # Debug line
        # print("First few rows:")
        # print(self.data.head())

    def upload_data(self, path):
        """
        Save data to CSV file.
        
        Removes NaN values before saving to ensure clean data.
        
        Args:
            path (str): Path to save CSV file.
        """
        self.remove_nans()
        self.data.to_csv(path, index=False)
    
    def generate_angle(self, gripper_pos, object_pos):
        """
        Calculate gripper orientation angles to point towards an object.
        
        Computes roll, pitch, and yaw angles such that the gripper points
        from an approach position (slightly above gripper) towards the object,
        with a small offset to account for optimal approach angle.
        
        Args:
            gripper_pos (numpy.ndarray): Gripper position [x, y, z].
            object_pos (list): Object position [x, y, z].
            
        Returns:
            numpy.ndarray: Array of [roll, pitch, yaw] angles in radians.
        """
        # Calculate approach position (slightly above gripper)
        approach_pos = gripper_pos + np.array([0, 0, 0.1])
        off = 0.04  # Offset distance for optimal approach angle (tunable parameter)
        
        # Calculate horizontal direction vector from object to approach position
        direction = np.array(approach_pos) - np.array(object_pos)
        direction[2] = 0  # Only horizontal offset (no vertical component)

        # Normalize direction and apply offset
        if np.linalg.norm(direction) > 0:  # Avoid division by zero
            direction = direction / np.linalg.norm(direction)  # Normalize
            offset = direction * off  # Scale by offset distance
        else:
            offset = np.array([0, 0, 0])

        # Calculate target position with offset
        target_pos = np.array(object_pos) + offset
        
        # Calculate direction from approach position to target
        direction = np.array(target_pos) - np.array(approach_pos)
        direction = direction / np.linalg.norm(direction)  # Normalize

        # Convert direction vector to Euler angles
        pitch = np.arcsin(-direction[2])  # Vertical angle
        yaw = np.arctan2(direction[1], direction[0])  # Horizontal angle
        roll = np.pi  # Fixed roll angle

        return np.array([roll, pitch, yaw])

    def update_success(self, idx, success=True):
        """
        Update success status for a specific data point.
        
        Args:
            idx (int): Index of the data point to update.
            success (bool, optional): Success status. Defaults to True.
        """
        self.data.at[idx, "success"] = success

    def remove_nans(self):
        """
        Remove rows with NaN values from the dataset.
        
        Modifies the DataFrame in place.
        """
        self.data.dropna(inplace=True)

    def statistics(self):
        """
        Print statistics about grasp success rates.
        
        Displays the count of successful vs failed grasps in the dataset.
        """
        self.remove_nans()
        print(self.data["success"].value_counts())

    def create_model_datasets(self, num_points, validation_points, test_points, shuffle=False):
        """
        Split data into balanced train/validation/test sets.
        
        Creates balanced datasets by taking equal numbers of success and failure
        cases from the data. Ensures class balance across all splits.
        
        Args:
            num_points (int): Number of training points (will be split 50/50 success/failure).
            validation_points (int): Number of validation points (will be split 50/50).
            test_points (int): Number of test points (will be split 50/50).
            
        Returns:
            tuple: (train_data, val_data, test_data) as pandas DataFrames.
            
        Raises:
            ValueError: If there are not enough data points to create the requested splits.
        """
        # Check if enough data is available
        required_points = (num_points/2 + validation_points/2 + test_points/2)
        if len(self.data) < required_points:
            raise ValueError(f"Not enough data points to create datasets. You only have {len(self.data)} points, but you need at least {required_points}.")
        
        self.data = self.data.sample(frac=1).reset_index(drop=True) if shuffle else self.data

        # Split into success and failure sets
        success_data = self.data[self.data["success"] == True].reset_index(drop=True)
        failure_data = self.data[self.data["success"] == False].reset_index(drop=True)

        train_data = pd.DataFrame()
        val_data = pd.DataFrame()
        test_data = pd.DataFrame()

        # TRAIN: Take equal numbers from success and failure sets
        for i in range(num_points//2):
            train_data = pd.concat([train_data,
                                    success_data.iloc[[i]],
                                    failure_data.iloc[[i]]],
                                axis=0)
        
        # VALIDATION: Continue from where training left off
        for i in range(num_points//2, num_points//2 + validation_points//2):
            val_data = pd.concat([val_data,
                                success_data.iloc[[i]],
                                failure_data.iloc[[i]]],
                                axis=0)

        # TEST: Continue from where validation left off
        start_idx = num_points//2 + validation_points//2
        end_idx = start_idx + test_points//2
        for i in range(start_idx, end_idx-1):
            test_data = pd.concat([test_data,
                                success_data.iloc[[i]],
                                failure_data.iloc[[i]]],
                                axis=0)

        # Reset indices and return
        return train_data.reset_index(drop=True), \
            val_data.reset_index(drop=True), \
            test_data.reset_index(drop=True)

    def visualise_data(self, file_name="output.jpg"):
        """
        Create a 3D visualization of gripper positions and orientations.
        
        Generates a 3D plot showing:
        - Gripper positions as points in space
        - Gripper orientations as arrows (quivers)
        - Success/failure color-coded (green=success, red=failure)
        - Reference cube at origin
        
        Args:
            file_name (str, optional): Filename to save the plot. Defaults to "output.jpg".
        """
        self.remove_nans()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        directions = []
        
        # Convert Euler angles to direction vectors for visualization
        for i, (roll, pitch, yaw) in enumerate(self.data[["roll", "pitch", "yaw"]].values):
            # Convert roll, pitch, yaw to unit direction vector
            dx = np.cos(yaw) * np.cos(pitch)
            dy = np.sin(yaw) * np.cos(pitch)
            dz = np.sin(pitch)
            directions.append([dx, dy, -dz])  # Negate dz for correct visualization

        directions = np.array(directions)

        # Color-code by success: green for success, red for failure
        colours = ["green" if c else "red" for c in self.data["success"]]

        # Draw quiver plot (arrows showing gripper orientation)
        ax.quiver(
            self.data["x"], self.data["y"], self.data["z"],
            directions[:, 0], directions[:, 1], directions[:, 2],
            length=0.1, normalize=True, color=colours, label="Orientation"
        )

        # Draw reference cube at origin
        cube_side = 0.06
        self.draw_reference_cube(ax, (0, 0, cube_side/2), cube_side)

        # Set axis labels and title
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Gripper Positions around a Reference Cube')
        ax.set_box_aspect([1, 1, 1])  # Equal scaling for x, y, z
        
        # Set axis limits
        ax.set_zlim(0, .8)
        ax.set_xlim(-.4, .4)
        ax.set_ylim(-.4, .4)
        
        # Save and display
        plt.savefig(file_name)
        plt.show()

    def draw_reference_cube(self, ax, center, side):
        """
        Draw a reference cube in the 3D plot.
        
        Draws a semi-transparent cube to represent the object being grasped,
        providing visual context for gripper positions.
        
        Args:
            ax: Matplotlib 3D axes object.
            center (tuple): Center position of cube (x, y, z).
            side (float): Edge length of the cube.
        """
        x, y, z = center
        r = [-side / 2, side / 2]  # Half-side length for vertex calculation

        # Generate all 8 vertices of the cube
        vertices = np.array([[x+dx, y+dy, z+dz] for dx in r for dy in r for dz in r])

        # Define 6 faces (each is a list of 4 vertices forming a quadrilateral)
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],  # bottom face
            [vertices[4], vertices[5], vertices[7], vertices[6]],  # top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back face
            [vertices[0], vertices[2], vertices[6], vertices[4]],  # left face
            [vertices[1], vertices[3], vertices[7], vertices[5]]   # right face
        ]

        # Create 3D polygon collection for the cube
        cube = Poly3DCollection(
            faces,
            alpha=0.3,          # Semi-transparent
            edgecolor='black',  # Black edges
            facecolor='orange'  # Orange faces
        )
        ax.add_collection3d(cube)
