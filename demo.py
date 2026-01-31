import pygame
import os 
from common import Simulator, STATE_DEMO, WIDTH, HEIGHT

if __name__ == "__main__":
    print("=========================================")
    print("   SELF PARKING CAR - DEMO MODE")
    print("   Loading best trained model...")
    print("=========================================")
    
    pygame.init()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SELF PARKING CAR SIMULATOR - DEMO")
    
    # Initialize simulator in Demo Mode
    sim = Simulator(SCREEN, mode=STATE_DEMO)
    
    # Try to load the best model first
    model_to_load = "Q_parking_model_best.pkl"
    if not os.path.exists(model_to_load):
        model_to_load = "Q_parking_model.pkl"
        
    print(f"Attempting to load: {model_to_load}")
    try:
        sim.brain.load(model_to_load)
        print("✓ Model successfully loaded.")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        print("Starting with untrained model.")

    print("\nControls:")
    print("- SPACE: Pause/Resume")
    print("- EXIT button: Close simulator")
    print("=========================================\n")
        
    sim.run()
