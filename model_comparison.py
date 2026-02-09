import pygame
import numpy as np
import random
import math
import csv
import os
from common import Simulator, STATE_DEMO, QBrain

# Ensure we don't need a window
os.environ["SDL_VIDEODRIVER"] = "dummy"

class RandomAgent:
    """
    A baseline agent that selects actions uniformly at random.
    This serves as a lower bound for performance.
    """
    def __init__(self):
        self.name = "Random Agent"
        self.eps = 0.0

    def get_action(self, state, demo=True):
        # 7 possible actions in the current environment
        return random.randint(0, 6)

    def load(self, path=None, verbose=False):
        pass

    def save(self, path=None, verbose=False):
        pass

    def learn(self, s, a, r, sn):
        pass

    def decay_params(self, rate):
        pass

    def save_best(self, verbose=True):
        pass

class HeuristicAgent:
    """
    A rule-based agent that uses simple logic to navigate towards the target.
    This serves as a competitive baseline to show if Q-learning learned useful strategies.
    """
    def __init__(self, simulator):
        self.name = "Heuristic Agent"
        self.sim = simulator
        self.eps = 0.0

    def get_action(self, state, demo=True):
        """
        Simple logic:
        1. Calculate angle to target.
        2. If aligned, move forward.
        3. If not aligned, turn towards target.
        4. Basic obstacle avoidance using LIDAR.
        """
        # Unpack state if needed, but we can access sim directly for heuristic
        # State: (xb, yb, ab, aeb, vb, (obs...))
        
        car = self.sim.car
        target = self.sim.target
        
        # Vector to target
        dx = target.centerx - car.pos.x
        dy = target.centery - car.pos.y
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        car_angle = car.angle % 360
        
        diff = (target_angle - car_angle + 180) % 360 - 180
        
        # LIDAR check (simple)
        # 0: Front, 1: Rear, 2: Left, 3: Right (indices from common.py get_observation)
        # Actually common.py: for i in [0, 4, 2, 6] -> Front, Back, Left, Right
        # But let's just use the state passed in if it has lidar data
        # state structure: (xb, yb, ab, aeb, vb, tuple(obs))
        lidar = state[5] 
        front_blocked = lidar[0]
        
        # Action Map:
        # 0: Idle, 1: Fwd, 2: Fwd-Left, 3: Fwd-Right, 
        # 4: Bwd, 5: Bwd-Left, 6: Bwd-Right
        
        if front_blocked:
            # If blocked ahead, try to reverse
            return 4 # Backward
            
        if abs(diff) < 10:
            return 1 # Forward
        elif diff > 0:
            return 3 # Fwd-Right (steer right to increase angle)
        else:
            return 2 # Fwd-Left (steer left to decrease angle)

    def load(self, path=None, verbose=False):
        pass

    def save(self, path=None, verbose=False):
        pass

    def learn(self, s, a, r, sn):
        pass

    def decay_params(self, rate):
        pass

    def save_best(self, verbose=True):
        pass

def run_comparison():
    print("Initializing Simulation for Model Comparison...")
    pygame.init()
    screen = pygame.Surface((1, 1)) # Dummy screen

    # Configuration
    EPISODES_PER_MODEL = 100
    results_file = "model_comparison_results.csv"

    # 1. Setup Models
    # We use STATE_TRAIN to get reward calculations, but we will disable learning/saving
    from common import STATE_TRAIN
    sim = Simulator(screen, mode=STATE_TRAIN, headless=True)
    
    # Disable Learning and Saving (Monkey Patching) to ensure fair comparison without side effects
    sim.brain.learn = lambda s, a, r, sn: None
    sim.brain.save = lambda path=None, verbose=True: None
    sim.brain.save_best = lambda verbose=True: None
    sim.brain.decay_params = lambda rate: None
    
    # Q-Learning Agent (The Star)
    q_agent = QBrain()
    # Explicitly load the best model
    if os.path.exists("Q_parking_model_best.pkl"):
        q_agent.load("Q_parking_model_best.pkl", verbose=True)
    else:
        print("Warning: Q_parking_model_best.pkl not found, using default/current Q_parking_model.pkl")
        q_agent.load("Q_parking_model.pkl", verbose=True)
    
    # Important: Set epsilon to 0 for greedy evaluation (best performance)
    q_agent.eps = 0.0
    
    # Random Agent
    random_agent = RandomAgent()
    
    # Heuristic Agent
    heuristic_agent = HeuristicAgent(sim)

    models = [
        ("Random Agent", random_agent),
        ("Heuristic Agent", heuristic_agent),
        ("Q-Learning Agent", q_agent)
    ]

    results = []

    print(f"\nStarting Comparison ({EPISODES_PER_MODEL} episodes each)...")

    for model_name, agent in models:
        print(f"\nTesting {model_name}...")
        
        # Swap brain
        if model_name == "Q-Learning Agent":
            sim.brain = q_agent
            # Re-apply patches just in case (though q_agent is a new object if not careful, 
            # but here we used the one we created. Wait, sim.brain was initialized in Simulator.
            # We are replacing it. So we need to patch the NEW brain if it's q_agent)
            sim.brain.learn = lambda s, a, r, sn: None
            sim.brain.save = lambda path=None, verbose=True: None
            sim.brain.save_best = lambda verbose=True: None
            sim.brain.decay_params = lambda rate: None
            sim.brain.eps = 0.0 # Ensure greedy
        else:
            sim.brain = agent
        
        successes = 0
        collisions = 0
        timeouts = 0
        total_rewards = 0
        total_steps = 0
        
        # Run episodes
        for ep in range(EPISODES_PER_MODEL):
            sim.reset_env()
            ep_reward = 0
            ep_steps = 0
            done = False
            
            # We need to capture the accumulated reward from the simulator
            # Simulator.ep_reward is reset in reset_env.
            # It is updated in step() -> calculate_reward().
            
            while not done:
                # Step the simulation
                outcome = sim.step()
                ep_steps += 1 # Manually count steps
                
                if outcome:
                    done = True
                    # Capture reward just before it might be reset (though reset happens inside step)
                    # Actually, reset_env is called INSIDE step if outcome is not None.
                    # So sim.ep_reward is ALREADY 0 here!
                    # We need to capture the reward *accumulation* or rely on the log.
                    # But we can't easily access the log of the just-finished episode.
                    
                    # Workaround: The return value of step() is the result.
                    # The reward was added to sim.ep_reward before reset.
                    # We missed it because it was reset.
                    
                    # We can use the Simulator's logger to get the last logged reward!
                    # sim.logger.reward_history[-1] should have it.
                    if sim.logger.reward_history:
                        ep_reward = sim.logger.reward_history[-1]
                    else:
                        ep_reward = 0
                    
                    if outcome == "Success":
                        successes += 1
                    elif outcome == "Collision":
                        collisions += 1
                    elif outcome == "Timeout":
                        timeouts += 1
            
            total_rewards += ep_reward
            total_steps += ep_steps
            
            if (ep + 1) % 10 == 0:
                print(f"  Ep {ep + 1}/{EPISODES_PER_MODEL}: {outcome} (Rew: {ep_reward:.1f})")

        # Aggregate metrics
        avg_reward = total_rewards / EPISODES_PER_MODEL
        avg_steps = total_steps / EPISODES_PER_MODEL
        success_rate = (successes / EPISODES_PER_MODEL) * 100
        
        results.append({
            "Model": model_name,
            "Success Rate (%)": success_rate,
            "Collision Rate (%)": (collisions / EPISODES_PER_MODEL) * 100,
            "Timeout Rate (%)": (timeouts / EPISODES_PER_MODEL) * 100,
            "Avg Reward": avg_reward,
            "Avg Steps": avg_steps
        })
        
        print(f"  > Result: {success_rate:.1f}% Success, Avg Reward: {avg_reward:.1f}")

    # Save results to CSV
    print(f"\nSaving results to {results_file}...")
    with open(results_file, 'w', newline='') as f:
        fieldnames = ["Model", "Success Rate (%)", "Collision Rate (%)", "Timeout Rate (%)", "Avg Reward", "Avg Steps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print("Comparison Complete!")

if __name__ == "__main__":
    run_comparison()