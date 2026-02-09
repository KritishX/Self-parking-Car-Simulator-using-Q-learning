# Self-Parking Car (SPC) Simulator

An advanced Reinforcement Learning (RL) project using **Tabular Q-Learning** to train an autonomous agent to perform precision parking. The model consistently achieves a stable **93%+ success rate** in rigorous validation tests, navigating complex parking lots with obstacles and non-holonomic vehicle constraints.

## 📁 Project Architecture

The project is modularized into several components to separate simulation logic, training automation, and comparative evaluation.

### Core Logic
- **`common.py`**: The heart of the project. Contains the `Simulator`, `AutonomousVehicle`, and `QBrain` classes. It handles the physics (Ackermann steering), state space discretization, LIDAR simulation, and the reward function.
- **`Q_parking_model_best.pkl`**: The persistent memory of the AI, containing the learned Q-Table (state-action pairs) for the highest performing model.
- **`Q_parking_model.pkl`**: The latest saved model from the training process.

### Training & Evaluation
- **`auto_train_test.py`**: A robust training orchestrator that runs iterative loops of training (3,000 episodes) followed by validation (100 episodes). Features automated hyperparameter tuning and "Fine-Tuning" phase triggers.
- **`fast_train.py`**: A high-speed, headless training script designed for continuous refinement of the best model using a moderate exploration rate.
- **`training.py`**: A simple entry point for visual training on the current model.
- **`demo.py` / `fast_demo.py`**: Entry points for visualizing or stress-testing (headless) the trained model without further learning.

### Benchmarking & Visualization
- **`model_comparison.py`**: A scientific validation script that benchmarks the Q-Learning Agent against a **Random Agent** (stochastic baseline) and a **Heuristic Agent** (rule-based baseline).
- **`model_visualization.py`**: Generates analytical charts from comparison results using Pandas and Seaborn, stored in the `/visualization` folder.

---

## 🧠 State Space Representation

To make the problem solvable via Tabular Q-Learning, the continuous environment is discretized into approximately **18,816 possible states**.

| Component | Discretization | Purpose |
| :--- | :--- | :--- |
| **X Zones** | 7 Bins | Relative horizontal distance to target center. |
| **Y Zones** | 7 Bins | Relative vertical distance to target center. |
| **Angle** | 12 Sectors | Car's orientation in 30° increments. |
| **Alignment** | 2 Bins | Boolean check: Is the car aligned within 10° of the slot? |
| **Speed** | 3 Bins | Moving Forward, Moving Backward, or Stopped (|v| < 0.2). |
| **LIDAR** | 8-Ray Focus | Binary "hit" sensors (Front, Back, Left, Right) within a 65px radius. |

---

## 💰 Reward & Penalty System

The model is guided by a **Dense Reward Function**, ensuring constant feedback for progress.

### Positive Rewards
- **Distance Potential (+250.0 per Δpx)**: Rewarded proportionally for every pixel moved closer to the target center.
- **Alignment Potential (+150.0 per Δdeg)**: When within 200px of the slot, the car earns rewards for rotating toward the perfect 90° or 270° angle.
- **Success Bonus (+20,000,000)**: A massive terminal reward for stopping inside the target with < 0.2 speed and < 15° alignment error.

### Penalties
- **Collision (-2,000,000)**: A heavy penalty for hitting walls or other parked cars, terminating the episode.
- **Timeout (-500,000)**: Penalizes failure to park within 2,000 steps.
- **Living Penalty (-15.0 per step)**: Encourages efficiency and minimizes path length.

---

## 📈 Training Methodology

The AI uses **Experience Replay** (batch size 32) and an automated **Curriculum Learning** loop. The `QBrain` employs a success-rate-aware parameter decay to balance exploration and exploitation:

- **Adaptive Epsilon (ε) Decay**:
  - **Success > 70%**: Rapid decay (0.99) to lock in mastery.
  - **Success 40-70%**: Moderate decay (0.995).
  - **Success 20-40%**: Slow decay (0.998).
  - **Success < 20%**: Very slow decay (0.9992) to encourage exploration.
- **Curriculum Phases**:
  1.  **Bootstrap (< 50% Success)**: No obstacles, car starts very close to the target.
  2.  **Intermediate (50-80% Success)**: Low obstacle probability, medium start distance.
  3.  **Advanced (80-95% Success)**: Obstacles present, wide starting variations.
  4.  **Mastery (> 95% Success)**: High obstacle density and extreme starting angles.

### Ultra Fine-Tuning Phase
Once the model hits **85% accuracy**, `auto_train_test.py` triggers a polish phase:
- **Alpha (Learning Rate)**: Dropped to **0.01** to stabilize the Q-Table.
- **Epsilon (Exploration)**: Forced to **0.01** to minimize random mistakes.

---

## 🏎️ Vehicle Physics

The simulation uses an **Ackermann Steering (Bicycle) Model**:
- **Turning Radius**: Calculated based on the `WHEELBASE` (70px) and steering angle.
- **Dynamics**: Includes momentum, acceleration (0.6), and friction-based braking (0.92 friction factor).
- **Action Space**: 7 core moves (Idle, Forward, Forward-Left, Forward-Right, Backward, Backward-Left, Backward-Right).

---

## 🚀 How to Run

### Requirements
```bash
pip install pygame numpy pandas seaborn tqdm
```

### Visual Demo
To see the best model in action:
```bash
python demo.py
```

### Run Benchmarks
To compare the AI against Random and Heuristic baselines:
```bash
python model_comparison.py
python model_visualization.py
```

### Train from Scratch
To run the full automated training cycle:
```bash
python auto_train_test.py
```