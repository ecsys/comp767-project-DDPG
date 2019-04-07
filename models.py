import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hyperparameters import *
# hyperparameters


class Actor(nn.Module):
    def __init__(self,input_size,output_size):
        super(Actor,self).__init__()
        self.linear1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE1)
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYER_SIZE1)
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2)
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYER_SIZE2)
        self.linear3 = nn.Linear(HIDDEN_LAYER_SIZE2, output_size)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
    def forward(self, state):
        h1 = self.linear1(state)
        # h1norm = self.bn1(h1)
        h2 = self.linear2(self.relu(h1))
        h2 = self.relu(h2)
        # h2norm = self.bn2(h2)
        out = self.linear3(h2)
        return out

class Critic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Critic,self).__init__()
        self.linear1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE1)
        self.bn1 = nn.BatchNorm1d(HIDDEN_LAYER_SIZE1)
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2)
        self.bn2 = nn.BatchNorm1d(HIDDEN_LAYER_SIZE2)
        self.linear3 = nn.Linear(HIDDEN_LAYER_SIZE2, output_size)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
    def forward(self, state, action):
        x = torch.cat([state,action],1)
        h1 = self.linear1(x)
        # h1norm = self.bn1(h1)
        h2 = self.linear2(self.relu(h1))
        h2 = self.relu(h2)
        # h2norm = self.bn2(h2)
        out = self.linear3(h2)
        return out