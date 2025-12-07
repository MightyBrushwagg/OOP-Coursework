# Pipeline Design: Robotic Grasping Simulation & ML Training

## Overview

This project implements an end-to-end pipeline for robotic grasping research, combining **physics simulation** with **machine learning** to predict grasp success. The system uses PyBullet for realistic physics simulation and scikit-learn for training predictive models.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              MAIN ENTRY POINT                           │
│                                 main.py                                 │
│                    (CLI for simulation & training modes)                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│      MODE: "run"          │     │      MODE: "train"        │
│   Physics Simulation      │     │    ML Model Training      │
└───────────────┬───────────┘     └───────────────┬───────────┘
                │                                 │
                ▼                                 ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│       Simulation          │     │         Models            │
│  ┌─────────┐ ┌─────────┐  │     │  ┌─────────────────────┐  │
│  │ Gripper │ │ Object  │  │     │  │ Logistic Regression │  │
│  └─────────┘ └─────────┘  │     │  │ SVM                 │  │
│         │         │       │     │  │ Random Forest       │  │
│         └────┬────┘       │     │  └─────────────────────┘  │
│              ▼            │     └───────────────┬───────────┘
│       Data Collection     │                     │
└───────────────┬───────────┘                     │
                │                                 │
                ▼                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                          Data                               │
│              (CSV storage & preprocessing)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Simulation Module (`Simulation/simulation.py`)

The `Simulation` class orchestrates the entire data collection process:

**Responsibilities:**
- Initializes the PyBullet physics environment
- Creates scenes with grippers and target objects
- Executes grasp-and-lift sequences
- Monitors contact points to determine success/failure
- Collects position, orientation, and outcome data

**Key Process Flow:**
```
1. start_simulation()       → Initialize PyBullet engine
2. create_scene()           → Load gripper + object at specified pose
3. gripper.grasp_and_lift() → Execute FULL grasp sequence (approach → close → lift)
4. Hold verification        → Monitor if grasp is sustained at lifted height
5. Record result            → Store success/failure in Data object
6. reset_scene()            → Clean up for next iteration
```

**Two-Phase Execution:**

| Phase | Description | Duration |
|-------|-------------|----------|
| **Grasp & Lift** | `grasp_and_lift()` executes entirely: approach object, close gripper, lift to target height (0.4m) | ~550 simulation steps |
| **Hold Verification** | Monitor if object remains grasped at the lifted height | Up to 2000 steps |

> **Note:** Success verification only begins **after** the gripper has completed the lift sequence. This tests the gripper's ability to **sustain** a grasp, not whether the lift itself succeeded. Objects that slip mid-lift will still reach the verification phase but fail the hold test.

**Success Criteria (evaluated during hold phase):**
- ≥2 contact points between gripper and object
- 0 contact points between object and ground plane
- Maintained for 720 consecutive simulation steps (~3 seconds at 240Hz)

---

### 2. Gripper Hierarchy (`Grippers/grippers.py`)

An inheritance-based design using the **Template Method Pattern**:

```
Gripper (Abstract Base Class)
    │
    ├── TwoFingerGripper (PR2 gripper)
    │       └── Simple parallel-jaw gripper
    │
    └── NewGripper (Robotiq 2F-85)
            └── Advanced gripper with mimic joints
```

**Base Class (`Gripper`)** provides:
- URDF loading and PyBullet body management
- Fixed constraint attachment for programmatic movement
- Position/orientation control via constraint updates
- Camera tracking for visualization

**Concrete Classes** implement:
- `start()` - Initialize joint positions
- `open()` / `close()` - Finger actuation
- `generate_angles()` - Calculate approach orientation
- `target_position()` - Compute grasp target with offset
- `grasp_and_lift()` - Full grasp sequence (approach → close → lift)

---

### 3. Object Hierarchy (`Objects/objects.py`)

Simple inheritance for graspable objects:

```
SceneObject (Abstract Base Class)
    │
    ├── Box (cube_small.urdf)
    │       └── grasp_height = 0.03m
    │
    └── Cylinder (cylinder.urdf)
            └── grasp_height = 0.1m
```

Each object defines a `grasp_height` attribute that tells the gripper the optimal z-coordinate for grasping.

---

### 4. Data Module (`Data/data.py`)

The `Data` class handles all data operations:

**Data Generation:**
- Generates random gripper positions on a sphere around the object
- Adds Gaussian noise for realistic distribution
- Calculates appropriate orientation angles for each position
- Filters points below ground level

**Storage Format (DataFrame columns):**
| Column | Description |
|--------|-------------|
| x, y, z | Gripper position coordinates |
| roll, pitch, yaw | Gripper orientation (Euler angles) |
| success | Boolean: grasp outcome |

**ML Data Preparation:**
- `create_model_datasets()` splits data into balanced train/validation/test sets
- Ensures equal representation of success/failure cases

**Visualization:**
- 3D scatter plot with orientation vectors (quivers)
- Color-coded by success (green) / failure (red)
- Reference cube at origin for context

---

### 5. Models Module (`Models/models.py`)

Implements the **Strategy Pattern** for interchangeable ML algorithms:

```
Model (Abstract Base Class)
    │
    ├── Logistic_Regression
    │       └── Binary classification baseline
    │
    ├── SVM (Support Vector Machine)
    │       └── Non-linear decision boundaries
    │
    └── Random_Forest
            └── Ensemble method for robustness
```

**Common Interface:**
- `fit()` - Train on training data
- `validate()` - Evaluate on validation set
- `test()` - Final evaluation on test set
- `predict()` - Inference on new data

**Feature Vector:** `[x, y, z, roll, pitch, yaw]` → 6-dimensional input

**Utility Function:**
`compare_models()` trains all models and returns accuracy comparison dictionary.

---

## Data Flow Pipeline

### Phase 1: Data Collection (Simulation)

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Generate    │    │   Execute    │    │   Hold       │    │   Record     │
│  Random      │───▶│   Grasp &    │───▶│   Verify     │───▶│   Outcome    │
│  Positions   │    │   Lift       │    │   (3 sec)    │    │   (CSV)      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                   │                   │
      ▼                    ▼                   ▼                   ▼
 Data.make_data()    grasp_and_lift()    Monitor contacts    update_success()
                     (full sequence)      at lifted height
```

**Timeline for single iteration:**
```
├─── grasp_and_lift() ───┤├────── Hold Verification ──────┤
     ~550 steps                    720+ steps required
     (approach,close,lift)         (to confirm success)
```

### Phase 2: Model Training

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Load CSV    │    │   Split &    │    │   Train &    │
│  Data        │───▶│   Balance    │───▶│   Evaluate   │
│              │    │   Dataset    │    │   Model      │
└──────────────┘    └──────────────┘    └──────────────┘
      │                    │                   │
      ▼                    ▼                   ▼
   Data.import_data()   create_model_datasets()   Model.fit() / test()
```

---

## Design Patterns Used

| Pattern | Application |
|---------|-------------|
| **Template Method** | `Gripper` base class defines grasp workflow; subclasses implement specifics |
| **Strategy** | `Model` classes are interchangeable algorithms with same interface |
| **Factory** | `Simulation.obj_dic` and `gripper_dic` map strings to class constructors |
| **Composition** | `Simulation` contains `Data`, `Gripper`, and `SceneObject` instances |

---

## Command-Line Interface

```bash
# Run 1000 simulation iterations with cube and two-finger gripper
python main.py --mode run --object cube --gripper two_finger --iterations 1000

# Train logistic regression on collected data
python main.py --mode train --model logistic_regression --file_save cube-two_finger-data.csv

# Compare all models
python main.py --mode train --model all --train_points 120 --test_points 300
```

---

## Key Design Decisions

1. **Abstraction via Inheritance**: Base classes (`Gripper`, `SceneObject`, `Model`) define interfaces, enabling easy extension with new grippers, objects, or ML algorithms.

2. **Separation of Concerns**: Each module has a single responsibility:
   - `Simulation` → Physics orchestration
   - `Data` → Data management
   - `Models` → ML training/inference
   - `Grippers/Objects` → Entity definitions

3. **Configurable Pipeline**: Command-line arguments allow flexible experimentation without code changes.

4. **Balanced Datasets**: The `create_model_datasets()` method ensures class balance, critical for training reliable classifiers.

5. **Dual Visualization Modes**: Simulations can run with GUI (`visuals`) for debugging or headless (`no visuals`) for fast batch data collection.

