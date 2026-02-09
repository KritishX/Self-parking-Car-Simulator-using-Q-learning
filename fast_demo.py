import os
import pygame
import csv
from datetime import datetime
from common import Simulator, STATE_DEMO, WIDTH, HEIGHT

def run_fast_demo(episodes=100, model_path="Q_parking_model_best.pkl"):
    print("=========================================")
    print(f"   FAST DEMO MODE (HEADLESS EVALUATION)")
    print(f"   Testing: {model_path}")
    print(f"   Episodes: {episodes}")
    print("=========================================")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    dummy_screen = pygame.Surface((1, 1))

    # Initialize simulator in Demo Mode
    sim = Simulator(dummy_screen, mode=STATE_DEMO, headless=True)
    
    # Load the specified model
    try:
        sim.brain.load(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Results tracking
    results = {"Success": 0, "Collision": 0, "Timeout": 0}
    
    # Setup CSV logging for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"evaluation_log_{timestamp}.csv"
    
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Outcome", "Steps", "Dist_Error", "Ang_Error"])
        
        for i in range(1, episodes + 1):
            outcome = None
            # Run one episode
            for res in sim.run():
                outcome = res
                break # sim.run() in headless yields once per episode completion
            
            results[outcome] += 1
            
            # Get errors from the simulator's last state
            dist_err = sim.car.pos.distance_to(sim.target.center)
            ang_err = sim.calculate_angle_error()
            
            writer.writerow([i, outcome, sim.steps, round(dist_err, 2), round(ang_err, 2)])
            
            if i % 10 == 0:
                success_rate = (results["Success"] / i) * 100
                print(f"  Processed {i}/{episodes} | Current Success Rate: {success_rate:.1f}%")

    # Final Summary
    total = sum(results.values())
    success_rate = (results["Success"] / total) * 100
    collision_rate = (results["Collision"] / total) * 100
    timeout_rate = (results["Timeout"] / total) * 100

    print("\n=========================================")
    print("   EVALUATION COMPLETE")
    print(f"   Total Episodes: {total}")
    print(f"   Success Rate:   {success_rate:.2f}%")
    print(f"   Collision Rate: {collision_rate:.2f}%")
    print(f"   Timeout Rate:   {timeout_rate:.2f}%")
    print(f"   Results saved to: {log_filename}")
    print("=========================================\n")

    pygame.quit()

if __name__ == "__main__":
    import sys
    eps = 100
    if len(sys.argv) > 1:
        eps = int(sys.argv[1])
    
    run_fast_demo(episodes=eps)
