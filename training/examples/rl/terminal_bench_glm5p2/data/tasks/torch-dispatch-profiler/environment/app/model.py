"""Target model for profiling.

A feedforward network with intentionally sub-optimal matmul shapes:
- fc1: Linear(512, 768)  -- well-aligned shapes (both divisible by 32)
- fc2: Linear(768, 3073) -- N=3073 is odd, not cache-aligned
- fc3: Linear(3073, 10)  -- K=3073 and N=10, both misaligned
"""
import torch
import torch.nn as nn


class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 768, bias=False)
        self.fc2 = nn.Linear(768, 3073, bias=False)
        self.fc3 = nn.Linear(3073, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
