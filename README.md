# Robotic Grasping Simulation and ML Training Pipeline

A PyBullet-based simulation framework for robotic grasping research, combining physics simulation with machine learning to predict grasp success. This project was developed as part of an Object-Oriented Programming coursework.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Simulations](#running-simulations)
  - [Training Models](#training-models)
  - [Testing Models](#testing-models)
- [Architecture](#architecture)
- [Data Format](#data-format)
- [Available Components](#available-components)

## Overview

This project implements an end-to-end pipeline for robotic grasping research:

1. **Simulation**: Uses PyBullet physics engine to simulate grasp attempts with different grippers and objects
2. **Data Collection**: Records gripper positions, orientations, and grasp outcomes
3. **Model Training**: Trains machine learning classifiers to predict grasp success
4. **Evaluation**: Tests trained models on new data and generates confusion matrices

The simulation generates random gripper poses around a target object, executes grasp-and-lift sequences, and determines success based on whether the gripper can maintain contact with the object while lifting it off the ground.

## Requirements

- Python 3.8+
- PyBullet
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MightyBrushwagg/OOP-Coursework.git
cd OOP-Coursework
```

2. Install dependencies:
```bash
pip install pybullet numpy pandas matplotlib scikit-learn joblib
```

## Usage

The project is operated through the command-line interface in `main.py`. There are three modes: `run` (simulation), `train` (model training), and `test` (model evaluation).

### Running Simulations

Generate grasping data by running physics simulations:

```bash
# Basic simulation with default settings (1000 iterations, cube, two-finger gripper)
python main.py --mode run

# Simulation with visual output
python main.py --mode run --visuals "visuals" --iterations 100

# Simulation with cylinder object and new gripper
python main.py --mode run --object cylinder --gripper new_gripper --iterations 500

# Run without saving data (for testing)
python main.py --mode run --save_data False --visuals "visuals"
```

**Simulation Parameters:**

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--object` | `cube`, `cylinder` | `cube` | Target object type |
| `--gripper` | `two_finger`, `new_gripper` | `two_finger` | Gripper type |
| `--visuals` | `visuals`, `no visuals` | `no visuals` | GUI or headless mode |
| `--iterations` | integer | 1000 | Number of grasp attempts |
| `--save_data` | `True`, `False` | `True` | Save results to CSV |
| `--file_save` | string | auto-generated | Output filename |

### Training Models

Train machine learning models on collected data:

```bash
# Train logistic regression model
python main.py --mode train --model logistic_regression --object cube --gripper two_finger

# Train Random Forest with custom parameters
python main.py --mode train --model forest --n_estimators 200 --train_points 150 --test_points 100

# Train SVM model
python main.py --mode train --model svm --object cylinder --gripper new_gripper

# Compare all models
python main.py --mode train --model all --train_points 120 --test_points 300
```

**Training Parameters:**

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `logistic_regression`, `svm`, `forest`, `all` | `logistic_regression` | Model type |
| `--train_points` | integer | 120 | Training set size |
| `--test_points` | integer | 300 | Test set size |
| `--val_points` | integer | 0 | Validation set size |
| `--n_estimators` | integer | 100 | Trees in Random Forest |
| `--shuffle` | `True`, `False` | `True` | Shuffle data before splitting |

### Testing Models

Load and evaluate pre-trained models:

```bash
# Test a trained Random Forest model
python main.py --mode test --model forest --object cube --gripper two_finger

# Test with custom data file
python main.py --mode test --model svm --file_save validation.csv
```

## Architecture

The project follows object-oriented design principles with the following class hierarchy:

**Simulation Package:**
- `Simulation` - Orchestrates PyBullet environment, manages grasp attempts, and records outcomes

**Grippers Package:**
- `Gripper` (abstract) - Base class defining common gripper behaviour
- `TwoFingerGripper` - PR2 parallel-jaw gripper implementation
- `NewGripper` - Robotiq 2F-85 gripper with mimic joints

**Objects Package:**
- `SceneObject` (abstract) - Base class for graspable objects
- `Box` - Cube object with grasp_height = 0.03m
- `Cylinder` - Cylinder object with grasp_height = 0.1m

**Models Package:**
- `Model` (abstract) - Base class defining ML model interface
- `Logistic_Regression` - Binary classification baseline
- `SVM` - Support Vector Machine classifier
- `Random_Forest` - Ensemble classifier

**Data Package:**
- `Data` - Handles data generation, import/export, preprocessing, and visualisation

See `UML_diagram.puml` for the complete class diagram.

## Data Format

Simulation data is stored in CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| x | float | Gripper x-coordinate |
| y | float | Gripper y-coordinate |
| z | float | Gripper z-coordinate |
| roll | float | Gripper roll angle (radians) |
| pitch | float | Gripper pitch angle (radians) |
| yaw | float | Gripper yaw angle (radians) |
| success | boolean | Whether the grasp was successful |

**Success Criteria:**
- At least 2 contact points between gripper and object
- No contact between object and ground plane
- Sustained for 720 simulation steps (approximately 3 seconds at 240Hz)

## Available Components

### Grippers

| Gripper | Description |
|---------|-------------|
| `two_finger` | PR2 parallel-jaw gripper (simple, fast) |
| `new_gripper` | Robotiq 2F-85 with mimic joint constraints |

### Objects

| Object | Description |
|--------|-------------|
| `cube` | 6cm cube using PyBullet's cube_small.urdf |
| `cylinder` | Custom cylinder with 10cm grasp height |

### Models

| Model | Description |
|-------|-------------|
| `logistic_regression` | Linear classifier, good baseline |
| `svm` | Support Vector Machine with RBF kernel |
| `forest` | Random Forest ensemble classifier |

## Outputs

The pipeline generates:

- **Data files**: `Data/<object>-<gripper>-data.csv`
- **Visualisation plots**: `Data/<object>-<gripper>-data.jpg`
- **Trained models**: `Models/saved_models/<object>_<gripper>_<model>_model.pkl`
- **Confusion matrices**: `Models/saved_models/<object>_<gripper>_<model>_confusion_matrix.jpg`
