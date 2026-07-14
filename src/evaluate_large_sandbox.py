import argparse
import os
import sys
import numpy as np
import torch
import math
import pygame

# Allow relative imports inside src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env import SafeNavigationEnv
from agent import ActorNetwork

class LargeSafeNavigationEnv(SafeNavigationEnv):
    """
    Subclass of SafeNavigationEnv configured with an expanded 800x800px grid
    and an increased count of dynamically generated obstacles.
    """
    def __init__(self, render_mode=None, num_obstacles=8):
        super().__init__(render_mode=render_mode)
        self.width, self.height = 800, 800
        self.num_obstacles = num_obstacles
        self.max_steps = 700  # Increased steps to allow traversal across the larger space

    def reset(self, seed=None, options=None):
        """Resets the environment with randomly generated obstacle distributions."""
        super(SafeNavigationEnv, self).reset(seed=seed)
        self.current_step = 0
        
        # Randomly generate obstacles mimicking the training distribution
        self.obstacles = []
        for i in range(self.num_obstacles):
            obs_x = self.np_random.uniform(200.0, 600.0)
            obs_y = self.np_random.uniform(50.0, 750.0)
            obs_r = self.np_random.uniform(25.0, 45.0)
            vy = self.np_random.uniform(-2.5, 2.5)
            if abs(vy) < 0.5: 
                vy = 1.5
            self.obstacles.append({"x": obs_x, "y": obs_y, "r": obs_r, "vy": vy})
            
        # Spawn agent safely in the left quadrant
        valid_spawn = False
        while not valid_spawn:
            self.agent_pos = np.array([
                self.np_random.uniform(50.0, 150.0),
                self.np_random.uniform(100.0, 700.0)
            ], dtype=np.float32)
            valid_spawn = True
            for obs in self.obstacles:
                dist = np.linalg.norm(self.agent_pos - np.array([obs["x"], obs["y"]]))
                if dist <= (obs["r"] + self.min_safe_distance + self.agent_radius + 5.0):
                    valid_spawn = False
                    break 

        self.agent_vel = 0.0
        self.agent_heading = self.np_random.uniform(-math.pi/4, math.pi/4)  
        self.prev_steer = 0.0
        
        # Spawn target in the right quadrant
        self.target_pos = np.array([
            self.np_random.uniform(650.0, 750.0),
            self.np_random.uniform(100.0, 700.0)
        ], dtype=np.float32)
        
        self.prev_lidar = self._get_lidar_readings()
        observation = self._get_obs()
        info = {"cost": 0.0}
        
        self.closest_dist_achieved = np.linalg.norm(self.target_pos - self.agent_pos)
        return observation, info

    def _get_obs(self):
        """Constructs vector states, clipping target bounds to the training distribution range."""
        rel_target = self.target_pos - self.agent_pos
        # Clip coordinates to training range to prevent out-of-distribution neural activation issues
        rel_target = np.clip(rel_target, -550.0, 550.0)
        
        vx = self.agent_vel * math.cos(self.agent_heading)
        vy = self.agent_vel * math.sin(self.agent_heading)
        lidar_feats = self._get_lidar_readings()
        
        lidar_change = lidar_feats - self.prev_lidar
        self.prev_lidar = lidar_feats
        
        obs = np.array([vx, vy, rel_target[0], rel_target[1]], dtype=np.float32)
        return np.concatenate([obs, lidar_feats, lidar_change])

    def step(self, action):
        """Executes a single physics step inside the sandbox."""
        self.current_step += 1
        accel, steer = action[0], action[1]
        
        # Update dynamic obstacles (wall bouncing)
        for obs in self.obstacles:
            obs["y"] += obs["vy"]
            if obs["y"] - obs["r"] <= 0 or obs["y"] + obs["r"] >= self.height:
                obs["vy"] *= -1.0

        # Update agent physics
        accel_modifier = 0.2 if accel >= 0 else 0.6
        self.agent_heading += steer * self.max_steering_v
        self.agent_vel = np.clip(self.agent_vel + accel * accel_modifier, 0.0, self.max_speed)
        
        self.agent_pos[0] += self.agent_vel * math.cos(self.agent_heading)
        self.agent_pos[1] += self.agent_vel * math.sin(self.agent_heading)
        
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.width)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.height)

        # Rewards progress
        dist_to_target = np.linalg.norm(self.target_pos - self.agent_pos)
        reward = 0.0
        
        if dist_to_target < self.closest_dist_achieved:
            progress_gain = self.closest_dist_achieved - dist_to_target
            reward += progress_gain * 3.0  
            self.closest_dist_achieved = dist_to_target
            
        reward -= 0.2 
        
        if dist_to_target <= self.closest_dist_achieved:
            reward += (self.agent_vel * 0.1)
        
        # Proximity braking penalty
        lidar_readings = self._get_lidar_readings()
        front_danger = min(lidar_readings[0], lidar_readings[1], lidar_readings[-1])
        
        if front_danger < 0.4: 
            if self.agent_vel > 1.0:
                reward -= self.agent_vel * 2.0 

        # Smooth steering control penalty
        steer_change = steer - self.prev_steer
        reward -= 0.15 * (steer_change ** 2)
        self.prev_steer = steer

        # Constraint cost calculation
        cost = 0.0
        collided = False
        
        for obs in self.obstacles:
            dist_to_obj = np.linalg.norm(self.agent_pos - np.array([obs["x"], obs["y"]])) - self.agent_radius - obs["r"]
            if dist_to_obj <= 0:
                collided = True
                cost += 10.0  
            elif dist_to_obj < self.min_safe_distance:
                ratio = 1.0 - (dist_to_obj / self.min_safe_distance)
                cost += 0.5 * (ratio ** 2)

        # Episode flags
        terminated = False
        if dist_to_target < (self.agent_radius + 10):
            reward += 50.0  
            terminated = True
        elif collided:
            reward -= 150.0
            terminated = True
            
        truncated = self.current_step >= self.max_steps
        obs_vec = self._get_obs()
        info = {"cost": cost, "collided": collided}

        if self.render_mode == "human":
            self._render_frame()

        return obs_vec, reward, terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot generalization in a large sandbox environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of generalization episodes to run")
    parser.add_argument("--obstacles", type=int, default=8, help="Number of dynamic obstacles in the environment")
    parser.add_argument("--weights-path", type=str, default="models/safe_sac_actor.pth", help="Path to trained actor weights")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running generalization evaluation on device: {device}...")
    print(f"Sandbox configuration: 800x800px Grid | {args.obstacles} Dynamic Obstacles")
    
    render_mode = "human" if args.render else None
    env = LargeSafeNavigationEnv(render_mode=render_mode, num_obstacles=args.obstacles)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = ActorNetwork(obs_dim, action_dim).to(device)
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at '{args.weights_path}'. Run training first.")
        env.close()
        return
        
    actor.load_state_dict(torch.load(args.weights_path, map_location=device))
    actor.eval()
    print(f"-> Successfully loaded weights from {args.weights_path}")
    
    success_count = 0
    collision_count = 0
    total_rewards = []
    total_costs = []
    
    print(f"\nEvaluating over {args.episodes} random test episodes...")
    for ep in range(1, args.episodes + 1):
        state, info = env.reset()
        ep_reward = 0
        ep_cost = 0
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = actor(state_t)
                action = torch.tanh(mean).cpu().squeeze(0).numpy()
                
            state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_cost += step_info["cost"]

            if args.render:
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
        
        status = "SUCCESS" if reached_goal else "COLLISION" if step_info.get("collided") else "TIMEOUT"
        if args.episodes <= 10 or ep % 10 == 0 or ep == args.episodes:
            print(f"Episode {ep:03d}/{args.episodes} | Reward: {ep_reward:.1f} | Cost: {ep_cost:.2f} | Status: {status}")
        
    success_rate = (success_count / args.episodes) * 100
    collision_rate = (collision_count / args.episodes) * 100
    
    print("\n" + "="*80)
    print("             MID-SIZED SANDBOX ZERO-SHOT GENERALIZATION SUMMARY")
    print("="*80)
    print(f"{'Success Rate':<28}: {success_rate:.1f}%")
    print(f"{'Collision Rate':<28}: {collision_rate:.1f}%")
    print(f"{'Average Episode Reward':<28}: {np.mean(total_rewards):.2f}")
    print(f"{'Average Episode Safety Cost':<28}: {np.mean(total_costs):.2f}")
    print("="*80)
    print(f"* Generalization score evaluated over a area size 1.77x larger with {args.obstacles} obstacles.")
    
    env.close()

if __name__ == "__main__":
    main(parse_args())
