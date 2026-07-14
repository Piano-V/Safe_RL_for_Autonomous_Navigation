import sys
import time
import pygame

def main():
    """Diagnostic utility verifying Pygame renderer works correctly on the display environment."""
    print("Testing Pygame window rendering context...")
    pygame.init()
    
    width, height = 600, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pygame Rendering Diagnostic")
    
    clock = pygame.time.Clock()
    
    # Run test loop for 3 seconds
    start_time = time.time()
    while time.time() - start_time < 3.0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        screen.fill((240, 240, 240))
        # Draw target diagnostic circle
        pygame.draw.circle(screen, (41, 128, 185), (300, 300), 50)
        
        pygame.display.update()
        clock.tick(30)
        
    pygame.quit()
    print("Pygame display context diagnostic test complete!")

if __name__ == "__main__":
    main()
