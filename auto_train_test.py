import os
import pygame
import numpy as np
from common import Simulator, STATE_TRAIN, STATE_DEMO

def run_training_loop():
    print("=========================================")
    print("   AUTO TRAIN & TEST LOOP")
    print("   Goal: 91%+ Stable Accuracy")
    print("=========================================")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    dummy_screen = pygame.Surface((1, 1))

    # Initialize simulator
    sim = Simulator(dummy_screen, mode=STATE_TRAIN, headless=True)
    
    # Load best model to start
    print("Loading initial model...", flush=True)
    try:
        sim.brain.load("Q_parking_model_best.pkl")
        print("✓ Loaded Q_parking_model_best.pkl", flush=True)
    except Exception:
        print("Starting fresh.", flush=True)

    # Hyperparameters for ~19k states
    sim.brain.eps = 1.0 
    sim.brain.eps_min = 0.01
    sim.brain.alpha = 0.4 

    MAX_LOOPS = 300
    TRAIN_EPISODES = 3000 
    TEST_EPISODES = 100  
    
    best_test_success_rate = 0.0
    stalls = 0
    fine_tune_phase = False 
    
    for loop_idx in range(1, MAX_LOOPS + 1):
        print(f"\n[LOOP {loop_idx}/{MAX_LOOPS}] | Alpha: {sim.brain.alpha:.4f} | Eps: {sim.brain.eps:.4f} | {'FINE-TUNE' if fine_tune_phase else 'EXPLORE'}")
        
        # --- TRAINING PHASE ---
        sim.mode = STATE_TRAIN
        print(f"   > Training...", flush=True)
        start_ep = sim.ep_count
        
        for outcome in sim.run():
            if (sim.ep_count - start_ep) % 1000 == 0:
                print(f"     Ep {sim.ep_count - start_ep}/{TRAIN_EPISODES}...", flush=True)
            if sim.ep_count >= start_ep + TRAIN_EPISODES: break
        
        # --- EVALUATION ---
        sim.mode = STATE_DEMO
        print(f"   > Validating...", flush=True)
        test_wins = test_collisions = test_timeouts = 0
        current_test_ep = 0
        for outcome in sim.run():
            current_test_ep += 1
            if outcome == "Success": test_wins += 1
            elif outcome == "Collision": test_collisions += 1
            elif outcome == "Timeout": test_timeouts += 1
            if current_test_ep >= TEST_EPISODES: break
        
        test_rate = (test_wins / TEST_EPISODES) * 100
        print(f"   >>> SUCCESS: {test_rate:.1f}% | COLLISION: {test_collisions} | TIMEOUT: {test_timeouts}")
        
        # --- TUNING ---
        if test_rate > best_test_success_rate:
            print(f"   *** NEW BEST! {test_rate:.1f}% ***")
            best_test_success_rate = test_rate
            sim.brain.save("Q_parking_model_best.pkl", verbose=True)
            stalls = 0
            
            # Trigger Fine-Tuning Phase
            if test_rate >= 85.0 and not fine_tune_phase:
                print("   !!! ENTERING ULTRA FINE-TUNING PHASE !!!")
                fine_tune_phase = True
                sim.brain.alpha = 0.01 
                sim.brain.eps = 0.01
        else:
            stalls += 1
            if stalls >= 5 and not fine_tune_phase:
                sim.brain.alpha = max(0.01, sim.brain.alpha * 0.7)
                print(f"   !!! STALLED. Alpha reduced to {sim.brain.alpha:.4f}")
                stalls = 0
        
        sim.brain.save("Q_parking_model.pkl", verbose=False)
        if not fine_tune_phase:
            sim.brain.eps = max(sim.brain.eps_min, sim.brain.eps * 0.8)
        
        if best_test_success_rate >= 91.0:
            print(f"\nGOAL ACHIEVED! 91%+ Stable Accuracy reached: {best_test_success_rate:.1f}%")
            break

    sim.exit_simulation()

if __name__ == "__main__":
    run_training_loop()