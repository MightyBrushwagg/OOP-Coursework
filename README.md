# Robot Simulation & Planning Framework

This project provides a full pipeline for generating simulated robot data, training a classifier on that data, and testing grasping strategies inside a PyBullet-based simulation. It is structured around a single entry point (`main.py`) that exposes multiple modes via `argparse`.

## ✨ Features

* **PyBullet Grasp Execution** - Execute and test robotic grasping strategies with configurable objects and grippers in simulation.
* **Dataset Generation** - Automatically generate labeled grasp data from PyBullet robot simulations with success/failure outcomes.
* **Classifier Training** - Train ML models (Logistic Regression, SVM, Random Forest) to predict grasp success using the generated dataset.
* **Model Comparison** - Compare multiple models simultaneously to find the best performer for your grasp prediction task.
* **Modular Code Structure** - Components split across `simulation`, `data`, `models`, `Objects` and `Grippers` folders for clarity.

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your_repo_link>
cd <repo_name>
```

### 2. Create & activate a Python 3 environment
```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

This project requires Python 3 and the following major packages:

* `numpy<2` (PyBullet currently requires NumPy 1.x for compatibility)
* `pybullet`
* `scikit-learn` (for ML classifiers)

Install everything with:
```bash
pip install -r requirements.txt
```

If PyBullet fails to import due to NumPy 2 issues, downgrade NumPy explicitly:
```bash
pip install "numpy<2"
```

## 🚀 Usage

All functionality is accessed via:
```bash
python main.py --mode <mode> [additional arguments]
```

### Available Modes

#### 1. Run Simulation (Generate Dataset)
```bash
python main.py --mode run --object cube --gripper two_finger --iterations 1000
```

**Arguments**

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--object` | str | cube | cube, cylinder | Type of object to grasp |
| `--gripper` | str | two_finger | two_finger, new_gripper | Type of gripper to use |
| `--visuals` | str | no visuals | visuals, no visuals | Whether to show PyBullet GUI |
| `--iterations` | int | 1000 | - | Number of grasp attempts to simulate |
| `--save_data` | bool | True | - | Whether to save data to CSV |
| `--file_save` | str | None | - | Custom filename for saved data (optional) |

This runs the simulation and stores the output as `{object}-{gripper}-data.csv` in the project directory (or uses your custom filename).

**Example:**
```bash
# Generate 500 samples with cylinder and new gripper, show visuals
python main.py --mode run --object cylinder --gripper new_gripper --iterations 500 --visuals visuals
```

#### 2. Train Classifier
```bash
python main.py --mode train --model logistic_regression --file_save cube-two_finger-data.csv
```

**Arguments**

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--model` | str | logistic_regression | logistic_regression, svm, forest, all | Model type to train |
| `--file_save` | str | None | - | CSV file to load data from |
| `--train_points` | int | 120 | - | Number of training samples |
| `--test_points` | int | 300 | - | Number of testing samples |
| `--val_points` | int | 0 | - | Number of validation samples |
| `--n_estimators` | int | 100 | - | Number of trees (Random Forest only) |

**Available Models:**
- `logistic_regression` - Logistic Regression classifier
- `svm` - Support Vector Machine classifier
- `forest` - Random Forest classifier
- `all` - Train and compare all three models

**Examples:**
```bash
# Train a single model
python main.py --mode train --model svm --file_save cylinder-new_gripper-data.csv --train_points 150

# Compare all models
python main.py --mode train --model all --file_save cube-two_finger-data.csv

# Train Random Forest with 200 trees
python main.py --mode train --model forest --n_estimators 200
```

## 📊 Workflow Example

Here's a typical workflow for this project:
```bash
# Step 1: Generate training data with cube and two-finger gripper
python main.py --mode run --object cube --gripper two_finger --iterations 1000

# Step 2: Train and compare all models on the generated data
python main.py --mode train --model all --file_save cube-two_finger-data.csv

# Step 3: Test with a different object/gripper combination
python main.py --mode run --object cylinder --gripper new_gripper --iterations 800

# Step 4: Train best-performing model on new data
python main.py --mode train --model forest --file_save cylinder-new_gripper-data.csv
```

## 📁 Project Structure
```
├── main.py                    # Main entry point with argparse interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── simulation/               # Simulation logic and PyBullet interface
├── data/                     # Data processing and handling
├── models/                   # ML model implementations
│   ├── Logistic_Regression
│   ├── SVM
│   ├── Random_Forest
│   └── compare_models
├── Objects/                  # Object URDF/mesh files
└── Grippers/                 # Gripper URDF/mesh files
```

## 📝 Notes

* Ensure Python 3 is used when running the project.
* PyBullet is not yet fully compatible with NumPy 2.x. If you see the error: "numpy.core.multiarray failed to import" downgrade NumPy using:
```bash
pip install "numpy<2"
```

* Generated CSV files contain grasp pose data (x, y, z, roll, pitch, yaw) and success labels.
* The default data split is 120 training / 300 testing points, but this can be adjusted via command-line arguments.
* When using `--model all`, the system will output comparative accuracy metrics for all three classifiers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
