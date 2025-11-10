import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Data:
    def __init__(self, data=None, num_points=500,object_pos=[0,0,0]):
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

    def make_data(self, object_pos, num_points = 400, R = 0.5, height = 0):
        # Step 1: Generate random 3D Gaussian vectors
        coords = np.random.normal(size=(3, num_points))
        # Step 2: Calculate the magnitude of each vector
        distance_from_origin = np.linalg.norm(coords, ord=2, axis=0)
        # Step 3: Normalize the vectors
        normalised_coords = coords / distance_from_origin
        # Scale to desired radius
        points = R * normalised_coords
        # print(points.T.shape)
        # print(points)
        noise = np.random.normal(0, 0.2, size=(3, num_points))

        noisy = points + noise
        valid = np.array([[coord[0], coord[1], coord[2]] for coord in noisy.T if coord[2] >= height])
        print(valid.shape)
        # print(valid)
        # positions = valid
        # print(positions)
        
        orientations = np.array([self.generate_angle(position, object_pos) for position in valid])
        # print(orientations)
        return valid, orientations # Transpose to get shape (num_points, 3)
    
    def import_data(self, path):
        self.data = pd.read_csv(path)
        self.positions = pd.DataFrame()

    def upload_data(self, path):
        self.remove_nans()
        self.data.to_csv(path)
    
    def generate_angle(self, gripper_pos, object_pos):
        approach_pos = gripper_pos + np.array([0,0,0.1])
        off = 0.04 # Need to iterate on this value
        direction = np.array(approach_pos) - np.array(object_pos)
        direction[2] = 0 # No z term offset needed because offset doesn't change based on starting position 

        if np.linalg.norm(direction) > 0: # No zero lengths 
            direction = direction / np.linalg.norm(direction) # Normalize direction vector
            offset = direction * off # Scale by offset
        else:
            offset = np.array([0, 0, 0])

        # print(offset)

        target_pos = np.array(object_pos) + offset # Apply offset
        direction = np.array(target_pos) - np.array(approach_pos)
        
        direction = direction/np.linalg.norm(direction)

        pitch = np.arcsin(-direction[2])
        yaw = np.arctan2(direction[1], direction[0])
        roll = np.pi

        return np.array([roll, pitch, yaw])

    def add_data(self, *args):
        self.data.loc[len(self.data)] = args
        # print(self.data)

    def remove_nans(self):
        self.data.dropna(inplace=True)

    def statistics(self):
        self.remove_nans()
        print(self.data["success"].value_counts())

    def visualise_data(self):
        self.remove_nans()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        directions = []
        
        
        for i, (roll, pitch, yaw) in enumerate(self.data[["roll", "pitch", "yaw"]].values):
            # _, row = t
            # print(i, roll, pitch, yaw)
            # roll, pitch, yaw = row["roll"], row["pitch"], row["yaw"]

            dx = np.cos(yaw) * np.cos(pitch)
            dy = np.sin(yaw) * np.cos(pitch)
            dz = np.sin(pitch)
            directions.append([dx,dy,-dz])

        directions = np.array(directions)

        colours = ["green" if c else "red" for c in self.data["success"]]

        ax.quiver(
            self.data["x"], self.data["y"], self.data["z"],
            directions[:, 0], directions[:, 1], directions[:, 2],
            length=0.1, normalize=True, color=colours, label="Orientation"
        )

        cube_side = 0.06
        self.draw_reference_cube(ax, (0,0,cube_side/2), cube_side)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Gripper Positions around a Reference Cube')
        # ax.legend()
        ax.set_box_aspect([1, 1, 1])  # Equal scaling for x, y, z
        ax.set_zlim(0, .8)
        ax.set_xlim(-.4,.4)
        ax.set_ylim(-.4,.4)
        plt.savefig("output.jpg")
        plt.show()


    def draw_reference_cube(self, ax, center, side):
        """Draws a cube centered at `center` with edge length `side`."""
        x, y, z = center
        r = [-side / 2, side / 2]

        # All cube vertices
        vertices = np.array([[x+dx, y+dy, z+dz] for dx in r for dy in r for dz in r])

        # 6 faces (each is a list of 4 vertices)
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],  # bottom
            [vertices[4], vertices[5], vertices[7], vertices[6]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[2], vertices[6], vertices[4]],  # left
            [vertices[1], vertices[3], vertices[7], vertices[5]]   # right
        ]

        cube = Poly3DCollection(
            faces,
            alpha=0.3,
            edgecolor='black',
            facecolor='orange'
        )
        ax.add_collection3d(cube)