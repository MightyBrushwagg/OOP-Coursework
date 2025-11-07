import numpy as np
import pandas as pd

# ---------- Parameters ----------
num_grippers = 200
cube_pos = [0, 0, 0.03]

positions = pd.DataFrame(columns=list("xyz"))
orientations = pd.DataFrame(columns=list("123"))

def generate_uniform_points(num_points, R=1, height = 0):
    coords = np.random.normal(size=(3, num_points))
    distance_from_origin = np.linalg.norm(coords, ord=2, axis=0)
    normalized_coords = coords / distance_from_origin
    points = R * normalized_coords
    valid = np.array([[coord[0], coord[1], coord[2]] for coord in points.T if coord[2] >= height])
    return valid

def generate_legal_position(points):
    noise = np.random.normal(0, 0.01, size=(3,))
    point = points[np.random.randint(0, len(points))]
    return point + noise

def generate_angles(gripper_pos, object_pos):
    direction = np.array(object_pos) - np.array(gripper_pos)
    direction = direction/np.linalg.norm(direction)
    pitch = np.arcsin(-direction[2])
    yaw = np.arctan2(direction[1], direction[0])
    roll = np.pi
    return np.array([roll, pitch, yaw])

# Generate points
points = generate_uniform_points(1000, R=0.5, height=cube_pos[2])

# Generate positions without PyBullet
for i in range(num_grippers):
    noisy_pos = generate_legal_position(points)
    positions.loc[len(positions)] = noisy_pos
    noisy_euler = generate_angles(noisy_pos, cube_pos)
    orientations.loc[len(orientations)] = noisy_euler

# Print in a copy-pasteable format
print("Positions:")
print(positions.values.tolist())
print("\nOrientations:")
print(orientations.values.tolist())