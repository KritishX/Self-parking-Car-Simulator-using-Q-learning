# Self-Parking Car (SPC) Simulator

An advanced Reinforcement Learning (RL) project using **Tabular Q-Learning** to train an autonomous agent to perform precision parking. The model has achieved a stable **93%+ success rate** in rigorous validation tests.

## 📁 Project Architecture

The project is modularized into several key components to separate simulation logic, training automation, and visualization.

### Core Logic
- **`common.py`**: The heart of the project. Contains the `Simulator`, `AutonomousVehicle`, and `QBrain` classes. It handles the physics (Ackermann steering), the state space discretization, and the reward function.
- **`Q_parking_model_best.pkl`**: The persistent memory of the AI, containing the learned Q-Table (state-action pairs).

### Training & Evaluation
- **`auto_train_test.py`**: A robust training orchestrator that runs iterative loops of training (3,000 episodes) followed by rigorous evaluation (100 episodes). It features automated hyperparameter tuning and "Fine-Tuning" phase triggers.
- **`fast_train.py`**: A high-speed, headless training script using `tqdm` for maximum throughput.
- **`demo.py` / `fast_demo.py`**: Entry points for visualizing or stress-testing the trained model without further learning.

---

## 🧠 State Space Representation

To make the problem solvable via Tabular Q-Learning, the continuous environment is discretized into approximately **18,816 possible states**.

| Component | Discretization | Purpose |
| :--- | :--- | :--- |
| **X Zones** | 7 Bins | Relative horizontal distance to target (Far/Near/Center). |
| **Y Zones** | 7 Bins | Relative vertical distance to target (Far/Near/Center). |
| **Angle** | 12 Sectors | Car's orientation in 30° increments. |
| **Alignment** | 2 Bins | Boolean check: Is the car aligned within 10° of the slot? |
| **Speed** | 3 Bins | Moving Forward, Moving Backward, or Idle (< 0.2 speed). |
| **LIDAR** | 4 Rays (Binary) | Detects obstacles/walls in 4 directions within a 65px radius. |

---

## 💰 Reward & Penalty System

The model is guided by a **Potential-Based Reward Function**, which ensures that every movement has a measurable impact on the car's goal.

### Positive Rewards
- **Distance Potential (+250.0 per Δpx)**: The car is rewarded proportionally for every pixel it moves closer to the target center.
- **Alignment Potential (+150.0 per Δdeg)**: When within 200px of the slot, the car earns rewards for rotating toward the perfect 90° or 270° parking angle.
- **Success Bonus (+20,000,000)**: A massive terminal reward for stopping inside the target with < 0.2 speed and < 10° alignment error.

### Penalties
- **Collision (-2,000,000)**: A heavy penalty for hitting walls or other parked cars, terminating the episode immediately.
- **Timeout (-500,000)**: Penalizes the car if it fails to park within 2,000 steps, preventing infinite loops.
- **Living Penalty (-15.0 per step)**: A constant small penalty that encourages the AI to find the most efficient path and park quickly.

---

## 📈 Training Methodology

The AI goes through four distinct **Curriculum Phases** to build skills progressively:

1.  **Bootstrap (< 50% Success)**: No obstacles, car starts very close to the target. Focus: Learning to enter the slot.
2.  **Intermediate (50-80% Success)**: Small obstacle probability, medium start distance. Focus: Basic navigation.
3.  **Advanced (80-95% Success)**: Obstacles present, wide variety of starting positions and angles. Focus: Precision.
4.  **Mastery (> 95% Success)**: High obstacle density and extreme starting variations. Focus: Total reliability.

### Adaptive Epsilon (ε) Decay
The `QBrain` uses a success-rate-aware decay. As the car improves, the exploration rate (Epsilon) drops faster to "lock in" the best behaviors.
- **Low Success**: ε decays slowly (0.9992) to encourage exploration.
- **High Success (> 70%)**: ε decays rapidly (0.99) to prioritize exploitation of the learned Q-Table.

### Fine-Tuning Phase
Once the model hits **85% stable accuracy**, `auto_train_test.py` triggers an **Ultra Fine-Tuning Phase**:
- **Alpha (Learning Rate)**: Dropped from 0.4 to **0.01**. This ensures that new experiences don't "overwrite" established good logic, allowing for microscopic adjustments to reach 91%+.
- **Epsilon**: Forced to **0.01** to minimize random mistakes during the final polish.

---

## 🏎️ Vehicle Physics

The simulation uses an **Ackermann Steering Model**:
- **Turning Radius**: Calculated based on the `WHEELBASE` (70px) and the current steering angle.
- **Velocity Dynamics**: Includes acceleration, friction-based braking, and momentum.
- **Discrete Action Space**: 7 core moves (Idle, Forward, Forward-Left, Forward-Right, Backward, Backward-Left, Backward-Right).

---

## 🚀 How to Run

### Requirements
```bash
pip install pygame numpy pandas tqdm
```

### Visual Demo
To see the 93% accurate model in action:
```bash
python demo.py
```

### Automated Training
To run the full iterative training cycle:
```bash
python auto_train_test.py
```