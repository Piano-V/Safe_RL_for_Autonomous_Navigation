import torch
import numpy as np

class LagrangianMultiplier:
    """
    Handles dynamic optimization of safety constraint weights using a linear 
    update rule with projection limits to prevent numerical overflow/instability.
    """
    def __init__(self, init_value=0.0, lr=0.05, cost_limit=1.0, max_value=20.0):
        self.cost_limit = cost_limit
        self.lr = lr
        self.max_value = max_value
        self.value = float(init_value)

    def torch_value(self):
        """Returns the current multiplier value as a PyTorch FloatTensor."""
        return torch.tensor(self.value, dtype=torch.float32)

    def update(self, cumulative_cost):
        """Updates the multiplier lambda based on constraint violations in the episode."""
        self.value = float(np.clip(
            self.value + self.lr * (cumulative_cost - self.cost_limit), 
            0.0, 
            self.max_value
        ))