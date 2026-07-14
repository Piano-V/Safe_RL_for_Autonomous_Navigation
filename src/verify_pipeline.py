import torch
import numpy as np
from env import SafeNavigationEnv
from agent import ActorNetwork, CriticNetwork
from lagrangian import LagrangianMultiplier

def main():
    """Performs sanity integration check for neural models and safe environment."""
    print("Checking full pipeline components integration...")
    
    # 1. Initialize environment
    env = SafeNavigationEnv(render_mode=None)
    obs, info = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"-> Environment loaded. Observation space dimension: {obs_dim}, Action space dimension: {action_dim}")
    
    # 2. Instantiate PyTorch models
    actor = ActorNetwork(obs_dim, action_dim)
    critic = CriticNetwork(obs_dim, action_dim)
    print("-> PyTorch actor-critic neural networks instantiated successfully.")
    
    # 3. Test actor and critic forward pass
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action, log_prob = actor.sample_action(obs_tensor)
        q1, q2 = critic(obs_tensor, action)
    
    print(f"-> Actor Forward Pass: Success. Sampled action: {action.numpy()[0]}")
    print(f"-> Twin Critic Forward Pass: Success. Q1: {q1.item():.4f}, Q2: {q2.item():.4f}")
    
    # 4. Initialize Safety Lagrangian multiplier
    lagrangian = LagrangianMultiplier(init_value=0.1, cost_limit=0.01)
    print(f"-> Lagrangian Multiplier: Ready. Initial Lambda: {lagrangian.value}")
    
    # 5. Execute test step in environment
    action_np = action.squeeze(0).numpy()
    next_obs, reward, terminated, truncated, step_info = env.step(action_np)
    print("-> Environment step execution with model action: Success.")
    print("\nIntegration check complete! All pipeline layers are fully compatible.")

if __name__ == "__main__":
    main()