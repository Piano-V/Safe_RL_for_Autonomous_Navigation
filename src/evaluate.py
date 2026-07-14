import argparse
import os
import sys
import numpy as np
import torch
import pygame

# Allow relative imports inside src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env import SafeNavigationEnv
from agent import ActorNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Safe SAC Agent.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes to run")
    parser.add_argument("--weights-path", type=str, default="models/safe_sac_actor.pth", help="Path to trained actor weights")
    parser.add_argument("--no-render", action="store_true", help="Disable visual Pygame rendering")
    return parser.parse_args()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    render_mode = None if args.no_render else "human"
    print(f"Initializing evaluation. Render mode: {render_mode}. Device: {device}")
    
    env = SafeNavigationEnv(render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Instantiate Actor
    actor = ActorNetwork(obs_dim, action_dim).to(device)
    
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at '{args.weights_path}'. Run training first.")
        env.close()
        return
        
    actor.load_state_dict(torch.load(args.weights_path, map_location=device))
    actor.eval()
    print(f"Loaded trained policy weights from: {args.weights_path}")

    for ep in range(1, args.episodes + 1):
        state, info = env.reset()
        ep_reward = 0
        ep_cost = 0
        done = False
        
        print(f"\n--- Running Evaluation Episode {ep}/{args.episodes} ---")
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Extract deterministic action (reparameterize=False)
                mean, _ = actor(state_t)
                action = torch.tanh(mean).cpu().squeeze(0).numpy()

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            ep_reward += reward
            ep_cost += step_info["cost"]
            
            # Smoothly handle exit event when running windowed mode
            if not args.no_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        sys.exit()

        status = "SUCCESS" if np.linalg.norm(env.target_pos - env.agent_pos) < (env.agent_radius + 10) else "TIMEOUT/COLLISION"
        print(f"Episode {ep} Finished | Status: {status} | Reward: {ep_reward:.2f} | Safety Cost: {ep_cost:.2f}")

    env.close()
    print("\nEvaluation run finished.")

if __name__ == "__main__":
    evaluate(parse_args())