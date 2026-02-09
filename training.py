import pygame
from common import Simulator, STATE_TRAIN, WIDTH, HEIGHT

if __name__ == "__main__":
    print("Starting Training Mode...")
    pygame.init()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SELF PARKING CAR SIMULATOR - TRAINING")
    # Initialize simulator in Training Mode
    sim = Simulator(SCREEN, mode=STATE_TRAIN)
    sim.run()
