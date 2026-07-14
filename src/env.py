import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class SafeNavigationEnv(gym.Env):
    """
    Gymnasium environment for 2D autonomous navigation under safety constraints.
    The agent must reach a target position while avoiding moving dynamic obstacles.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, num_lidar_beams=16, max_steps=500):
        super(SafeNavigationEnv, self).__init__()
        
        self.render_mode = render_mode
        self.num_lidar_beams = num_lidar_beams
        self.max_steps = max_steps
        self.current_step = 0
        
        # Grid dimensions (pixels)
        self.width, self.height = 600, 600
        
        # Kinematic and sensor parameters
        self.agent_radius = 15
        self.max_speed = 3.2
        self.max_steering_v = 0.25  
        self.lidar_max_range = 150.0
        self.min_safe_distance = 25.0  
        
        # Dynamic obstacles configured as [x_start, y_start, radius, velocity_y]
        self.obstacle_config = [
            [200, 200, 40, 2.5],
            [400, 400, 50, -2.0],
            [300, 150, 45, 3.0],
            [150, 450, 35, -1.5]
        ]
        self.obstacles = [] 

        # Action space: [linear_acceleration, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )

        # Observation space: [vx, vy, dx_target, dy_target] + lidar_readings + lidar_change
        obs_dim = 4 + 2 * self.num_lidar_beams
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Resets the environment state, obstacle locations, and agent pose."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset obstacles
        self.obstacles = []
        for x, y, r, vy in self.obstacle_config:
            self.obstacles.append({"x": float(x), "y": float(y), "r": float(r), "vy": float(vy)})

        # Spawn agent safely away from obstacles
        valid_spawn = False
        while not valid_spawn:
            self.agent_pos = np.array([
                self.np_random.uniform(30.0, 120.0),
                self.np_random.uniform(50.0, 550.0)
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
        
        # Spawn target destination on the opposite side of the map
        self.target_pos = np.array([
            self.np_random.uniform(480.0, 570.0),
            self.np_random.uniform(50.0, 550.0)
        ], dtype=np.float32)
        
        self.prev_lidar = self._get_lidar_readings()
        observation = self._get_obs()
        info = {"cost": 0.0}
        
        if self.render_mode == "human":
            self._render_frame()
            
        self.closest_dist_achieved = np.linalg.norm(self.target_pos - self.agent_pos)
        return observation, info

    def _get_obs(self):
        """Constructs the vector state representation of the environment."""
        rel_target = self.target_pos - self.agent_pos
        vx = self.agent_vel * math.cos(self.agent_heading)
        vy = self.agent_vel * math.sin(self.agent_heading)
        lidar_feats = self._get_lidar_readings()
        
        # Lidar first derivative models relative velocity to dynamic obstacles
        lidar_change = lidar_feats - self.prev_lidar
        self.prev_lidar = lidar_feats
        
        obs = np.array([vx, vy, rel_target[0], rel_target[1]], dtype=np.float32)
        return np.concatenate([obs, lidar_feats, lidar_change])

    def _get_lidar_readings(self):
        """Simulates 2D range beams around the agent detecting circular obstacles."""
        readings = []
        angles = np.linspace(0, 2 * math.pi, self.num_lidar_beams, endpoint=False)
        
        for angle in angles:
            beam_angle = self.agent_heading + angle
            dx = math.cos(beam_angle)
            dy = math.sin(beam_angle)
            
            min_dist = self.lidar_max_range
            
            for obs in self.obstacles:
                to_obj = np.array([obs["x"] - self.agent_pos[0], obs["y"] - self.agent_pos[1]])
                ray_dir = np.array([dx, dy])
                projection = np.dot(to_obj, ray_dir)
                
                if projection > 0: 
                    closest_pt = self.agent_pos + projection * ray_dir
                    dist_to_center = np.linalg.norm(closest_pt - np.array([obs["x"], obs["y"]]))
                    
                    if dist_to_center < obs["r"]: 
                        chord_len = math.sqrt(obs["r"]**2 - dist_to_center**2)
                        hit_dist = projection - chord_len
                        if 0 < hit_dist < min_dist:
                            min_dist = hit_dist
            readings.append(min_dist / self.lidar_max_range) 
            
        return np.array(readings, dtype=np.float32)

    def step(self, action):
        """Advances environment dynamics by one step given the agent's action."""
        self.current_step += 1
        accel, steer = action[0], action[1]
        
        # Update dynamic obstacle positions (vertical patrolling)
        for obs in self.obstacles:
            obs["y"] += obs["vy"]
            if obs["y"] - obs["r"] <= 0 or obs["y"] + obs["r"] >= self.height:
                obs["vy"] *= -1.0 

        # Apply action limits and dynamics
        accel_modifier = 0.2 if accel >= 0 else 0.6
        
        self.agent_heading += steer * self.max_steering_v
        self.agent_vel = np.clip(self.agent_vel + accel * accel_modifier, 0.0, self.max_speed)
        
        self.agent_pos[0] += self.agent_vel * math.cos(self.agent_heading)
        self.agent_pos[1] += self.agent_vel * math.sin(self.agent_heading)
        
        self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.width)
        self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.height)

        # Reward formulation
        dist_to_target = np.linalg.norm(self.target_pos - self.agent_pos)
        reward = 0.0
        
        # Reward positive distance progress
        if dist_to_target < self.closest_dist_achieved:
            progress_gain = self.closest_dist_achieved - dist_to_target
            reward += progress_gain * 3.0  
            self.closest_dist_achieved = dist_to_target
            
        # Step cost to optimize target path time
        reward -= 0.2 
        
        if dist_to_target <= self.closest_dist_achieved:
            reward += (self.agent_vel * 0.1)
        
        # Brake penalty: penalize forward momentum near obstacles
        lidar_readings = self._get_lidar_readings()
        front_danger = min(lidar_readings[0], lidar_readings[1], lidar_readings[-1])
        
        if front_danger < 0.4: 
            if self.agent_vel > 1.0:
                reward -= self.agent_vel * 2.0 

        # Jerk penalty: smooth control actions
        steer_change = steer - self.prev_steer
        reward -= 0.15 * (steer_change ** 2)
        self.prev_steer = steer

        # Constraint cost formulation
        cost = 0.0
        collided = False
        
        for obs in self.obstacles:
            dist_to_obj = np.linalg.norm(self.agent_pos - np.array([obs["x"], obs["y"]])) - self.agent_radius - obs["r"]
            if dist_to_obj <= 0:
                collided = True
                cost += 10.0  
            elif dist_to_obj < self.min_safe_distance:
                # Continuous penalty scaling up inside the safety buffer boundary
                ratio = 1.0 - (dist_to_obj / self.min_safe_distance)
                cost += 0.5 * (ratio ** 2)

        # Episode termination flags
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

    def _render_frame(self):
        """Handles visual rendering using Pygame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Dynamic Safe RL Autonomous Navigation")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((240, 240, 240)) 

        # Draw target destination (green circle)
        pygame.draw.circle(canvas, (46, 204, 113), self.target_pos.astype(int), 20)

        # Draw obstacles (red filled circle) and safety zones (hollow circle)
        for obs in self.obstacles:
            pos_int = (int(obs["x"]), int(obs["y"]))
            pygame.draw.circle(canvas, (231, 76, 60), pos_int, int(obs["r"]))
            pygame.draw.circle(canvas, (231, 76, 60), pos_int, int(obs["r"] + self.min_safe_distance), 1)

        # Draw robot agent (blue circle)
        pygame.draw.circle(canvas, (41, 128, 185), self.agent_pos.astype(int), self.agent_radius)
        
        # Draw heading direction vector (dark line)
        end_pt = self.agent_pos + np.array([math.cos(self.agent_heading), math.sin(self.agent_heading)]) * 25
        pygame.draw.line(canvas, (52, 73, 94), self.agent_pos.astype(int), end_pt.astype(int), 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Safely shuts down environment rendering resources."""
        if self.window is not None:
            pygame.quit()