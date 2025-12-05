Grasp Simulation, Dataset Generation, and Classifier Training
This project simulates robotic grasps using PyBullet and trains machine-learning models to classify grasp success.
It supports:
Dataset generation using different objects and grippers
Training multiple classifiers (Logistic Regression, SVM, Random Forest)
Testing / evaluation of trained models
The project uses a single entry point, main.py, with different modes selected via argparse.
📦 Features
✔ Dataset Generation
Simulates grasps with:
Objects: cube, cylinder
Grippers: two_finger, new_gripper
Each simulation run collects:
Object pose (x, y, z, roll, pitch, yaw)
Gripper configuration
Success label
Outputs a CSV file named:
<object>-<gripper>-data.csv
✔ Classifier Training
Three models are supported:
Logistic Regression
SVM
Random Forest
Each model is trained on the generated dataset and automatically evaluated.
⚙️ Installation
1. Clone the repository
git clone <your-repo-url>
cd <your-project-folder>
2. Install dependencies
This project depends on:
Python 3.9+
NumPy <2.0 (PyBullet compatibility)
PyBullet
scikit-learn
pandas
Install everything using:
pip install -r requirements.txt
If you run into PyBullet + NumPy issues on macOS, ensure:
pip install "numpy<2"
🚀 Running the Project
All functionality is accessed through main.py, using the --mode argument.
1️⃣ Generate a Dataset
python3 main.py --mode run --object <cube|cylinder> --gripper <two_finger|new_gripper>
All other arguments use defaults:
iterations = 1000
save_data = True
visuals = no visuals
Examples
python3 main.py --mode run --object cube --gripper two_finger
python3 main.py --mode run --object cube --gripper new_gripper
python3 main.py --mode run --object cylinder --gripper two_finger
python3 main.py --mode run --object cylinder --gripper new_gripper
Each command produces a CSV file:
cube-two_finger-data.csv
cube-new_gripper-data.csv
cylinder-two_finger-data.csv
cylinder-new_gripper-data.csv
2️⃣ Train a Classifier
You can train logistic regression, svm, or forest:
python3 main.py --mode train --model <logistic_regression|svm|forest> --object <obj> --gripper <grip>
Uses dataset:
<object>-<gripper>-data.csv
Examples
python3 main.py --mode train --model svm --object cube --gripper two_finger
python3 main.py --mode train --model logistic_regression --object cylinder --gripper new_gripper
python3 main.py --mode train --model forest --object cube --gripper new_gripper
3️⃣ (Optional) Testing a Planner
If you later implement a “planner test” mode, it should follow:
python3 main.py --mode test_planner --file_save <datafile>
Explain arguments here once implemented.
📁 File Structure
main.py
simulation.py
data.py
models.py
README.md
simulation.py – PyBullet simulation + data collection
data.py – CSV loading, preprocessing, statistics
models.py – ML classifiers (Logistic Regression, SVM, Random Forest)
main.py – argument parsing + mode selection
📝 Notes
All datasets are automatically saved to CSV when running in run mode.
Default simulation iteration count is 1000 unless changed with --iterations.
PyBullet requires NumPy <2.0 due to binary compatibility issues.