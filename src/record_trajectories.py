import argparse
import json
import os
import sys
import numpy as np
import torch

# Allow relative imports inside src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env import SafeNavigationEnv
from agent import ActorNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="Record evaluation trajectories of Safe SAC agent to a JSON file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of trajectories to record")
    parser.add_argument("--weights-path", type=str, default="models/safe_sac_actor.pth", help="Path to trained actor weights")
    parser.add_argument("--output-json", type=str, default="models/trajectories.json", help="Path to output JSON trajectory file")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SafeNavigationEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = ActorNetwork(obs_dim, action_dim).to(device)
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at '{args.weights_path}'. Run training first.")
        env.close()
        return
        
    actor.load_state_dict(torch.load(args.weights_path, map_location=device))
    actor.eval()
    
    trajectories = []
    
    print(f"Recording agent trajectories during {args.episodes} test episodes...")
    for ep in range(args.episodes):
        state, info = env.reset()
        done = False
        path = [env.agent_pos.tolist()]
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = actor(state_t)
                action = torch.tanh(mean).cpu().squeeze(0).numpy()
            
            state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            path.append(env.agent_pos.tolist())
            
        dist_to_target = np.linalg.norm(env.target_pos - env.agent_pos)
        success = bool(dist_to_target < (env.agent_radius + 10))
        
        trajectories.append({
            "episode": ep + 1,
            "start_pos": path[0],
            "target_pos": env.target_pos.tolist(),
            "path_coords": path,
            "success": success
        })
        print(f"-> Episode {ep + 1}/{args.episodes} recorded. Steps: {len(path)}. Success: {success}")
        
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"\nTrajectories successfully recorded and saved to: {args.output_json}")
    env.close()

if __name__ == "__main__":
    main(parse_args())
