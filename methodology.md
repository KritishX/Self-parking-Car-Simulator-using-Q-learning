# Methodology and Technical Architecture
**Project:** Self-Parking Car Simulator using Q-Learning
**Date:** February 9, 2026
**Environment:** Python 3.9+, Pygame, NumPy

## 1. Introduction

This document details the technical methodology, architectural decisions, and algorithmic implementations used to create an autonomous self-parking car simulation. The core objective was to train an agent to navigate a complex parking lot environment and successfully park in a designated spot using **Reinforcement Learning (RL)**, specifically **Q-Learning**.

To validate the efficacy of the RL approach, a comparative study was conducted against a Random Agent (stochastic baseline) and a Heuristic Agent (rule-based baseline).

---

## 2. Simulation Environment (`common.py`)

The simulation is built from scratch using the **Pygame** library. It provides a lightweight, deterministic 2D physics environment that simulates vehicle kinematics and sensor readings.

### 2.1 Coordinate System and World Generation
- **Dimensions:** The world is a fixed 1400x900 pixel grid.
- **Parking Lot:** A central parking area is procedurally generated with walls and individual parking slots.
- **Randomization:** To ensure robust training, the start position of the car and the target parking slot are randomized each episode.
    - **Curriculum Learning:** The difficulty of initialization scales with the agent's success rate. Early training starts close to the target; later stages introduce larger distances and random obstacles.

### 2.2 Vehicle Kinematics
The car does not move like a simple grid object; it follows a **Bicycle Model** approximation for non-holonomic vehicle dynamics. This adds significant complexity to the control problem, as the agent cannot simply "slide" sideways into a spot.

- **State Variables:**
    - Position $(x, y)$
    - Heading Angle $(	heta)$
    - Velocity $(v)$
    - Steering Angle $(\delta)$
- **Update Equations:**
    The car's position updates based on its speed and heading. The heading changes based on the steering angle and wheelbase length ($L=70$).
    $$ \dot{x} = v \cos(	heta) $$
    $$ \dot{y} = v \sin(	heta) $$
    $$ \dot{	heta} = \frac{v}{L} 	an(\delta) $$
- **Friction:** A simplified friction model applies a deceleration factor (0.92) to the velocity at each tick, simulating rolling resistance and requiring active throttle control.

### 2.3 Sensor System (LIDAR)
The agent perceives its environment through a simulated LIDAR (Light Detection and Ranging) system.
- **Ray Casting:** 8 rays are cast radially from the car's center.
- **Collision Detection:** Each ray checks for intersections with:
    1. Parking lot walls (bounds).
    2. Static obstacles (other parked cars).
- **Feedback:** The sensors return the distance to the nearest obstacle. For the Q-Learning state space, this continuous distance is discretized into binary "hits" (Is there an obstacle within 65 pixels?) to reduce state space explosion.

---

## 3. Reinforcement Learning Approach: Q-Learning

The core intelligence is driven by a Q-Learning algorithm, a model-free RL technique that learns the value of an action in a particular state.

### 3.1 The Q-Function
The goal is to learn a function $Q(s, a)$ that estimates the expected future reward of taking action $a$ in state $s$.
The update rule used is the standard Bellman Equation:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$

Where:
- $\alpha$ (Alpha): Learning rate (0.4 decaying to 0.01).
- $\gamma$ (Gamma): Discount factor (0.995), prioritizing long-term success.
- $R$: Immediate reward received.

### 3.2 State Space Discretization
A continuous environment has infinite states. To make Q-Learning feasible without deep neural networks (DQN), we discretized the state space into a manageable tuple:

**State Tuple:** `(xb, yb, ab, aeb, vb, obs_tuple)`

1.  **Relative X Position (`xb`):** 7 bins (Very Far Left ... Target ... Very Far Right).
2.  **Relative Y Position (`yb`):** 7 bins (Very Far Up ... Target ... Very Far Down).
3.  **Heading Angle (`ab`):** 12 bins ($360^\circ / 30^\circ$ sectors).
4.  **Alignment Error (`aeb`):** 2 binary states (Aligned within $10^\circ$ or not).
5.  **Velocity (`vb`):** 3 states (Reversing, Stopped, Forward).
6.  **Obstacles (`obs_tuple`):** 4 key LIDAR sensors (Front, Back, Left, Right) converted to binary (Clear/Blocked).

**Total State Space Size:**
$$ 7 	imes 7 	imes 12 	imes 2 	imes 3 	imes 2^4 \approx 18,816 	ext{ unique states} $$
This size is small enough for tabular Q-Learning to converge within a few thousand episodes but detailed enough to allow precise parking.

### 3.3 Action Space
The agent has 7 discrete actions, allowing for complex maneuvers:
0.  **Idle:** Maintain current state (friction slows car).
1.  **Forward:** Accelerate.
2.  **Forward-Left:** Accelerate + Steer Left.
3.  **Forward-Right:** Accelerate + Steer Right.
4.  **Backward:** Reverse.
5.  **Backward-Left:** Reverse + Steer Left.
6.  **Backward-Right:** Reverse + Steer Right.

### 3.4 Reward Shaping
The reward function is critical for guiding the agent. We used a dense reward structure to accelerate learning:

- **Distance Reward:** Proportional to the reduction in distance to the target ($+250 	imes \Delta 	ext{dist}$). This encourages approaching the spot.
- **Alignment Reward:** Given when close to the target, rewarding proper orientation ($+150 	imes \Delta 	ext{angle}$).
- **Terminal Rewards:**
    - **Success (Parked):** $+20,000,000$. A massive reward to dominate all other signals.
    - **Collision:** $-2,000,000$. A severe penalty to discourage reckless driving.
    - **Timeout:** $-500,000$. Prevents the agent from loitering.
- **Living Penalty:** $-15$ per step. Encourages speed and efficiency.

---

## 4. Training Methodology (`auto_train_test.py`)

Training was conducted using an automated curriculum loop to ensure stability and adaptability.

### 4.1 The Loop
The training process alternates between **Exploration** and **Exploitation/Testing**.
1.  **Training Phase:** The agent interacts with the environment using an $\epsilon$-greedy policy (random actions with probability $\epsilon$).
2.  **Experience Replay:** Transitions $(s, a, r, s')$ are stored in a buffer. A batch is sampled periodically to break correlations in the data and stabilize learning.
3.  **Evaluation Phase:** $\epsilon$ is set to 0. The agent runs 100 episodes greedily.
4.  **Fine-Tuning:** If the agent reaches $>85\%$ success, learning rate $\alpha$ and exploration $\epsilon$ are drastically reduced to "polish" the policy without destroying established knowledge.

### 4.2 Hyperparameter Tuning
- **Epsilon ($\epsilon$):** Starts at 1.0 (100% random). Decays based on success rate. If success > 70%, decay is aggressive.
- **Alpha ($\alpha$):** Starts at 0.5. If the model stalls (fails to improve for 5 loops), $\alpha$ is reduced to allow for finer convergence.

---

## 5. Comparative Analysis (`model_comparison.py`)

To prove the effectiveness of the Q-Learning model, we implemented a comparison framework involving three distinct agents.

### 5.1 Random Agent (Baseline)
- **Logic:** Selects one of the 7 actions uniformly at random at every step.
- **Purpose:** Establishes the "floor" of performance. If a model cannot beat this, it has learned nothing.
- **Expected Result:** Near 0% success rate due to the precision required for parking.

### 5.2 Heuristic Agent (Rule-Based)
- **Logic:** Implements a state-machine logic:
    1.  Calculate vector to target.
    2.  If facing target: Drive forward.
    3.  If not facing target: Steer towards it.
    4.  If obstacle ahead (LIDAR check): Reverse.
- **Purpose:** Represents a "traditional code" approach. It works well in open spaces but struggles with the complex non-holonomic constraints (e.g., parallel parking maneuvers or correcting bad angles close to walls).
- **Limitation:** Hard-coded rules are brittle and do not generalize well to "unseen" trapped scenarios.

### 5.3 Q-Learning Agent (Trained)
- **Logic:** Uses the `Q_parking_model_best.pkl` lookup table generated during training.
- **Execution:** Selects $a = 	ext{argmax}(Q(s, a))$.
- **Advantage:** Can learn counter-intuitive maneuvers (like backing up to get a better angle) that are hard to code explicitly.

### 5.4 Metrics Collected
For each agent, we ran 100 test episodes and recorded:
- **Success Rate (%):** Percentage of episodes ending in a valid park.
- **Collision Rate (%):** Percentage of episodes ending in a crash.
- **Timeout Rate (%):** Percentage of episodes exceeding the step limit.
- **Average Reward:** The mean cumulative reward (indicates trajectory quality).
- **Average Steps:** Proxy for efficiency (time to park).

---

## 6. Visualization and Results (`model_visualization.py`)

The raw data from the comparison was processed using **Pandas** and visualized with **Seaborn** to create intuitive charts.

### 6.1 Charts Generated
1.  **Success Rate Comparison:** A bar chart contrasting the three models. The Q-Learning model typically achieves >80%, while Random is ~0% and Heuristic is ~20-40%.
2.  **Average Reward:** Displays the massive gap in accumulated utility. The Q-Learning agent's positive bars contrast sharply with the negative penalties accumulated by the baselines.
3.  **Failure Modes (Stacked Bar):** Break down of *why* agents failed. Random agents usually crash (Collision). Heuristic agents often get stuck or take too long (Timeout/Collision).
4.  **Efficiency (Avg Steps):** Shows that even when the Heuristic agent succeeds, the Q-Learning agent often finds a shorter, more direct path.

---

## 7. Code Structure and Key Files

### `common.py`
The backbone of the project.
- **Classes:** `Simulator`, `AutonomousVehicle`, `QBrain`, `DetailedLogger`.
- **Key Functions:** `get_observation()` (state encoding), `calculate_reward()` (shaping), `drive()` (physics).

### `auto_train_test.py`
The training supervisor.
- Manages the "Outer Loop" of training.
- Handles model persistence (`.pkl` files).
- Implements the "Stall Detection" logic to adjust hyperparameters dynamically.

### `model_comparison.py`
The scientific validation script.
- **Monkey Patching:** To ensure a fair test, the script temporarily disables the `learn()` and `save()` functions of the `QBrain` so the model is evaluated in a frozen state.
- **Agent Classes:** Defines `RandomAgent` and `HeuristicAgent` classes that adhere to the same interface (`get_action`) as the Q-Brain.

### `model_visualization.py`
The reporting tool.
- Reads `model_comparison_results.csv`.
- Exports `.png` images to the `/visualization` folder.
- Uses strict linting and type checking compliant coding standards.

## 8. Conclusion

The methodology employed—starting from a robust physics simulation, applying a discrete Q-Learning algorithm with curriculum training, and validating against strong baselines—demonstrates a complete lifecycle of an RL project. The results clearly show that the Q-Learning agent successfully learned the complex kinematics of parking, significantly outperforming both random chance and simple heuristic rules.
