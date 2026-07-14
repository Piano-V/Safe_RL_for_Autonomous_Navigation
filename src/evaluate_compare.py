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
    parser = argparse.ArgumentParser(description="Compare Safe SAC (Lagrangian) and Unconstrained SAC (Static Penalty) policies.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of comparison episodes to run")
    parser.add_argument("--safe-weights", type=str, default="models/safe_sac_actor.pth", help="Path to Safe SAC model weights")
    parser.add_argument("--unconstrained-weights", type=str, default="models/unconstrained_sac_actor.pth", help="Path to Unconstrained SAC model weights")
    parser.add_argument("--render", action="store_true", help="Enable Pygame visual rendering")
    return parser.parse_args()

def evaluate_policy(actor, env, device, num_episodes, render):
    success_count = 0
    collision_count = 0
    total_rewards = []
    total_costs = []

    for ep in range(1, num_episodes + 1):
        state, info = env.reset()
        ep_reward = 0
        ep_cost = 0
        done = False
        
        actor_obs_dim = actor.fc1.in_features
        
        while not done:
            # Handle possible observation vector size discrepancies
            state_sliced = state[:actor_obs_dim]
            state_t = torch.FloatTensor(state_sliced).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = actor(state_t)
                action = torch.tanh(mean).cpu().squeeze(0).numpy()

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            ep_reward += reward
            ep_cost += step_info["cost"]

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        sys.exit()

        dist_to_target = np.linalg.norm(env.target_pos - env.agent_pos)
        reached_goal = dist_to_target < (env.agent_radius + 10)
        
        if reached_goal:
            success_count += 1
        if step_info.get("collided", False):
            collision_count += 1

        total_rewards.append(ep_reward)
        total_costs.append(ep_cost)

    success_rate = (success_count / num_episodes) * 100
    collision_rate = (collision_count / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)

    return success_rate, collision_rate, avg_reward, avg_cost

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comparative evaluation on device: {device}")

    render_mode = "human" if args.render else None
    env = SafeNavigationEnv(render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load Safe SAC Agent
    constrained_actor = ActorNetwork(obs_dim, action_dim).to(device)
    if os.path.exists(args.safe_weights):
        constrained_actor.load_state_dict(torch.load(args.safe_weights, map_location=device))
        constrained_actor.eval()
        print(f"-> Loaded Constrained Safe SAC Model from {args.safe_weights}")
    else:
        print(f"Warning: Safe model weights not found at '{args.safe_weights}'")
        constrained_actor = None

    # Load Unconstrained SAC Agent (checking observation dimension from file)
    if os.path.exists(args.unconstrained_weights):
        checkpoint = torch.load(args.unconstrained_weights, map_location=device)
        u_obs_dim = checkpoint["fc1.weight"].shape[1]
        unconstrained_actor = ActorNetwork(u_obs_dim, action_dim).to(device)
        unconstrained_actor.load_state_dict(checkpoint)
        unconstrained_actor.eval()
        print(f"-> Loaded Unconstrained SAC Model from {args.unconstrained_weights} (obs_dim: {u_obs_dim})")
    else:
        print(f"Warning: Unconstrained model weights not found at '{args.unconstrained_weights}'")
        unconstrained_actor = None

    if not constrained_actor and not unconstrained_actor:
        print("Error: No models found for evaluation. Exit.")
        env.close()
        return

    print(f"\nEvaluating policies over {args.episodes} episodes...")

    results = {}
    if constrained_actor:
        c_succ, c_coll, c_rew, c_cost = evaluate_policy(constrained_actor, env, device, args.episodes, args.render)
        results["Constrained SAC (Lagrangian)"] = (c_succ, c_coll, c_rew, c_cost)
    
    if unconstrained_actor:
        u_succ, u_coll, u_rew, u_cost = evaluate_policy(unconstrained_actor, env, device, args.episodes, args.render)
        results["Unconstrained SAC (Static Beta)"] = (u_succ, u_coll, u_rew, u_cost)

    print("\n" + "="*85)
    print("                              COMPARATIVE EVALUATION SUMMARY")
    print("="*85)
    print(f"{'Policy Model':<35} | {'Success Rate (%)':<18} | {'Collision Rate (%)':<20} | {'Avg Reward':<12} | {'Avg Safety Cost':<16}")
    print("-"*85)
    for model_name, metrics in results.items():
        succ, coll, rew, cost = metrics
        print(f"{model_name:<35} | {succ:<18.1f} | {coll:<20.1f} | {rew:<12.2f} | {cost:<16.2f}")
    print("="*85)
    print("\n* Note: An ideal safety policy balances success rate (high) with collision rate (0%).")
    
    env.close()

if __name__ == "__main__":
    main(parse_args())
