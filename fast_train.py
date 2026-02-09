import os
import pygame
from common import Simulator, STATE_TRAIN


if __name__ == "__main__":
    print("=========================================")
    print("   FAST TRAINING MODE (HEADLESS)")
    print("   Continuous Training - 30 Rounds")
    print("   Press Ctrl+C to save and exit")
    print("=========================================")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    dummy_screen = pygame.Surface((1, 1))

    # Initialize simulator once to preserve curriculum and memory
    sim = Simulator(dummy_screen, mode=STATE_TRAIN, headless=True)
    
    print("Loading best existing model for fine-tuning...", flush=True)
    try:
        sim.brain.load("Q_parking_model_best.pkl")
        print("✓ Loaded Q_parking_model_best.pkl", flush=True)
    except Exception as e:
        print(f"Could not load best model: {e}", flush=True)
        
    sim.brain.eps = 0.15      # moderate exploration to refine
    sim.brain.eps_min = 0.005 # lower floor
    sim.brain.alpha = 0.3     # more stable updates

    MAX_ROUNDS = 30
    EPISODES_PER_ROUND = 300
    TARGET_SUCCESS_RATE = 98.0
    
    overall_best_rate = 0.0

    try:
        for round_idx in range(1, MAX_ROUNDS + 1):
            print(f"\n--- ROUND {round_idx}/{MAX_ROUNDS} (Episodes {sim.ep_count} - {sim.ep_count + EPISODES_PER_ROUND}) ---", flush=True)
            
            round_results = {"Success": 0, "Collision": 0, "Timeout": 0}
            steps_in_round = 0
            
            # Run the batch
            start_ep = sim.ep_count
            for outcome in sim.run():
                round_results[outcome] += 1
                steps_in_round += 1
                
                # Update progress
                rate = (sum(sim.recent_successes) / len(sim.recent_successes) * 100) if sim.recent_successes else 0
                avg_rew = (sum(sim.logger.reward_history) / len(sim.logger.reward_history)) if sim.logger.reward_history else 0
                
                if steps_in_round % 20 == 0:
                     print(f"   Ep {steps_in_round}/{EPISODES_PER_ROUND} | Rate: {rate:.1f}% | Rew: {avg_rew:.1f} | Eps: {sim.brain.eps:.3f}", flush=True)

                if sim.ep_count >= start_ep + EPISODES_PER_ROUND:
                    break
            
            # End of Round Analysis
            total_round = sum(round_results.values())
            round_success_rate = (round_results["Success"] / total_round * 100) if total_round > 0 else 0
            current_curriculum_rate = (sum(sim.recent_successes) / len(sim.recent_successes) * 100) if sim.recent_successes else 0
            
            print(f"   Round Summary:")
            print(f"   - Success Rate (Batch): {round_success_rate:.2f}%")
            print(f"   - Curriculum Rate (Recent): {current_curriculum_rate:.2f}%")
            print(f"   - Epsilon: {sim.brain.eps:.4f}")
            print(f"   - Q-Table Size: {len(sim.brain.table)} states")

            # Save checkpoint
            sim.brain.save(verbose=False)
            
            if current_curriculum_rate > overall_best_rate:
                overall_best_rate = current_curriculum_rate
                print(f"   >>> NEW BEST RATE: {overall_best_rate:.2f}% <<<")
                sim.brain.save_best(verbose=False)

            # Check termination condition
            if current_curriculum_rate >= TARGET_SUCCESS_RATE:
                print(f"\n\nSUCCESS! Target accuracy of {TARGET_SUCCESS_RATE}% achieved!")
                break
                
            if round_idx == MAX_ROUNDS:
                print("\nMax rounds reached. Training Loop Completed.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("\nSaving final state and exiting...")
        sim.exit_simulation()