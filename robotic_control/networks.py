import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import *

class RewardModel(nn.Module):
	def __init__(self, env, n_hidden_layers=2, hidden_size=64):
		super().__init__()
		self.input_size = env.observation_space.shape[0]
		self.action_size = env.action_space.shape[0]
		self.layers = nn.ModuleList([nn.Linear(self.input_size + self.action_size, hidden_size)])
		self.layers.extend([nn.Linear(hidden_size, hidden_size) for i in range(n_hidden_layers-1)])
		self.out_layer = nn.Linear(hidden_size, 1)
		self.relu = nn.ReLU()	
	 
	def forward(self, x):
		for layer in self.layers:
			x = self.relu(layer(x))
		out = self.out_layer(x)
		return out
