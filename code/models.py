import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, list_size, hs):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(list_size, hs),
            nn.ReLU(),
            nn.Linear(hs, list_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, list_size, hs):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(list_size, hs),
            nn.ReLU(),
            nn.Linear(hs, 1)
        )
    
    def forward(self, state):
        return self.network(state)