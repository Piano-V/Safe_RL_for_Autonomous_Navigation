import os
import sys
import pygame

# Allow relative imports inside src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env import SafeNavigationEnv

def main():
    """Starts keyboard manual driving session for the custom environment."""
    env = SafeNavigationEnv(render_mode="human")
    obs, info = env.reset()
    
    print("\n======================================================================")
    print("Manual Control Mode Activated!")
    print("Drive the agent using Arrow Keys: [Up=Throttle, Down=Brake, Left/Right=Steer]")
    print("Press Escape or close the Pygame window to exit.")
    print("======================================================================\n")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        action = [0.0, 0.0]
        
        # Check active keyboard state
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:    
            action[0] = 0.5
        if keys[pygame.K_DOWN]:  
            action[0] = -0.5
        if keys[pygame.K_LEFT]:  
            action[1] = -0.5
        if keys[pygame.K_RIGHT]: 
            action[1] = 0.5
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info["cost"] > 0:
            print(f"⚠️ Proximity Breach / Collision detected! Step Cost: {info['cost']:.2f}")
            
        if terminated or truncated:
            print("🏳️ Target reached or session timeout. Resetting environment...")
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                
        clock.tick(30)
                
    env.close()
    print("Manual control session closed.")

if __name__ == "__main__":
    main()