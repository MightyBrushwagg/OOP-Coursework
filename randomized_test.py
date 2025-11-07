import pybullet as p
import pybullet_data
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- Simulation Setup ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

# ---------- Cube Object ----------
cube_pos = [0, 0, 0.03]
cube_id = p.loadURDF("cube_small.urdf", cube_pos)

p.resetDebugVisualizerCamera(
    cameraDistance=1,             # smaller = closer (try 0.2–0.5)
    cameraYaw=45,                   # rotate around cube
    cameraPitch=-30,                # tilt down
    cameraTargetPosition=cube_pos   # look directly at cube
)

# ---------- Parameters ----------
num_grippers = 200
base_position = np.array([0, 0, 0.5])
base_euler = np.array([0, np.pi/2, 0])  # 90° rotation along Y

prev_gripper_id = None

positions = pd.DataFrame(columns=list("xyz"))
print(positions)
orientations = pd.DataFrame(columns=list("123"))

def generate_uniform_points(num_points, R=1, height = 0):
    # Step 1: Generate random 3D Gaussian vectors
    coords = np.random.normal( size=(3, num_points))
    # Step 2: Calculate the magnitude of each vector
    distance_from_origin = np.linalg.norm(coords, ord=2, axis=0)
    # Step 3: Normalize the vectors
    normalized_coords = coords / distance_from_origin
    # Scale to desired radius
    points = R * normalized_coords
    # print(points.T.shape)
#     print(points)

    valid = np.array([[coord[0], coord[1], coord[2]] for coord in points.T if coord[2] >= height])
    # print(valid.shape)
    # print(valid)

    return valid # Transpose to get shape (num_points, 3)

# Example usage
points = generate_uniform_points(1000, R=0.5, height = cube_pos[2])
# ---------- Spawn grippers sequentially ----------
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

for i in range(num_grippers):
    # Remove previous gripper
    if prev_gripper_id is not None:
        p.removeBody(prev_gripper_id)


    # Add Gaussian noise
    noisy_pos = generate_legal_position(points) # use np.random.normal(mean, standard_deviation, size_of_output) 
    # noisy_pos = np.array([0,0,0.5])
    
    # print(noisy_pos)
    positions.loc[len(positions)] = noisy_pos
    # noisy_euler = base_euler + np.random.normal(0,np.pi/10, size=(3)) # use np.random.normal(mean, standard_deviation, size_of_output) 
    # noisy_euler = np.array([np.pi,0,0])
    noisy_euler = generate_angles(noisy_pos, cube_pos)
    # print(noisy_euler)
    noisy_quat = p.getQuaternionFromEuler(noisy_euler)
    # print(noisy_quat)
    orientations.loc[len(orientations)] = noisy_euler

    # Load gripper
    gripper_id = p.loadURDF("pr2_gripper.urdf", noisy_pos, noisy_quat)

    # Make gripper stay in place
    p.createConstraint(
        parentBodyUniqueId=gripper_id,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=noisy_pos,
        childFrameOrientation=noisy_quat
    )

    # Visualize briefly
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1./240.)

    prev_gripper_id = gripper_id

p.disconnect()

print(positions.values.tolist())
# print(positions)
# print(orientations)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter gripper positions
# ax.scatter(positions["x"], positions["y"], positions["z"], color='blue', s=40, label='Grippers')

# Draw orientation arrows (fixed the repeated “1”)
# Convert euler angles to direction vectors
directions = []
for _, row in orientations.iterrows():
    roll, pitch, yaw = orientations["1"], orientations["2"], orientations["3"]

    # Compute direction vector (the "forward" Z-axis of the rotated frame)
    dx = np.cos(yaw) * np.cos(pitch)
    dy = np.sin(yaw) * np.cos(pitch)
    dz = np.sin(pitch)
    directions.append([dx, dy, -dz])

directions = np.array(directions)
# print(directions)

# Now plot properly:
ax.quiver(
    positions["x"], positions["y"], positions["z"],
    directions[:, 0], directions[:, 1], directions[:, 2],
    length=0.1, normalize=True, color='red', label="Orientation"
)
# ax.quiver(
#     positions["x"], positions["y"], positions["z"],
#     orientations["2"], orientations["1"], orientations["3"],
#     length=0.1, normalize=True, color='red'
# )

# ---------- Draw reference cube at origin ----------
def draw_reference_cube(ax, center, side):
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

cube_side = 0.06
draw_reference_cube(ax, (0, 0, cube_side / 2), cube_side)

# ---------- Labels and aesthetics ----------
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
