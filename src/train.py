import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from env import SafeNavigationEnv
from agent import ActorNetwork, CriticNetwork, ReplayBuffer
from lagrangian import LagrangianMultiplier

def parse_args():
    parser = argparse.ArgumentParser(description="Train Safe SAC Agent with Lagrangian Constraint optimization.")
    parser.add_argument("--episodes", type=int, default=1200, help="Number of training episodes")
    parser.add_argument("--start-steps", type=int, default=3000, help="Steps taken before training starts")
    parser.add_argument("--warmup-episodes", type=int, default=150, help="Episodes before constraint updates begin")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for actor/critic optimizers")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="SAC temperature parameter")
    parser.add_argument("--cost-limit", type=float, default=1.0, help="Lagrangian constraint cost threshold")
    parser.add_argument("--lagrangian-lr", type=float, default=0.05, help="Learning rate for multiplier update")
    parser.add_argument("--save-path", type=str, default="models/safe_sac_actor.pth", help="Filepath to save the actor weights")
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Safe SAC Agent on device: {device}")

    env = SafeNavigationEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent networks
    actor = ActorNetwork(obs_dim, action_dim).to(device)
    critic = CriticNetwork(obs_dim, action_dim).to(device)
    target_critic = CriticNetwork(obs_dim, action_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr)

    lagrangian = LagrangianMultiplier(init_value=0.0, lr=args.lagrangian_lr, cost_limit=args.cost_limit)
    replay_buffer = ReplayBuffer(1000000)

    reward_history = []
    cost_history = []
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        done = False

        while not done:
            total_steps += 1

            # Select exploration or policy action
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_t, _ = actor.sample_action(state_t)
                    action = action_t.cpu().squeeze(0).numpy()

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            cost = step_info["cost"]

            replay_buffer.push(state, action, reward, next_state, cost, float(done))
            state = next_state
            episode_reward += reward
            episode_cost += cost

            # Skip optimization if experience replay has insufficient data
            if len(replay_buffer) < args.batch_size or total_steps < args.start_steps:
                continue

            states, actions, rewards, next_states, costs, dones = replay_buffer.sample(args.batch_size)

            states_t = torch.FloatTensor(states).to(device)
            actions_t = torch.FloatTensor(actions).to(device)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(next_states).to(device)
            costs_t = torch.FloatTensor(costs).unsqueeze(1).to(device)
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

            # Compute Target Q values
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample_action(next_states_t)
                target_q1, target_q2 = target_critic(next_states_t, next_actions)
                target_q = torch.min(target_q1, target_q2) - args.alpha * next_log_probs
                
                # Dynamic Lagrangian safety scaling
                if episode < args.warmup_episodes:
                    current_lambda = torch.tensor(0.0).to(device)
                else:
                    current_lambda = lagrangian.torch_value().to(device)
                
                # Lagrangian discounted reward
                backup = (rewards_t - current_lambda * costs_t) + (1.0 - dones_t) * args.gamma * target_q

            # Update critic networks
            current_q1, current_q2 = critic(states_t, actions_t)
            critic_loss = F.mse_loss(current_q1, backup) + F.mse_loss(current_q2, backup)
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update actor policy network
            sampled_actions, log_probs = actor.sample_action(states_t)
            q1, q2 = critic(states_t, sampled_actions)
            min_q = torch.min(q1, q2)
            
            actor_loss = (args.alpha * log_probs - min_q).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Target network soft update (Polyak)
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

        reward_history.append(episode_reward)
        cost_history.append(episode_cost)

        # Update Lagrangian multiplier
        if episode >= args.warmup_episodes:
            lagrangian.update(episode_cost)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            avg_cost = np.mean(cost_history[-10:])
            print(f"Episode: {episode:04d}/{args.episodes} | Avg Reward: {avg_reward:.1f} | Avg Cost: {avg_cost:.2f} | Lambda: {lagrangian.value:.4f}")

    print(f"\nTraining Complete! Saving actor weights to: {args.save_path}")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(actor.state_dict(), args.save_path)

if __name__ == "__main__":
    train(parse_args())