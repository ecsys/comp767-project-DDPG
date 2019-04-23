import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hyperparameters import *
# hyperparameters


    # def normalize_state(self, state):
    #     # normalize
    #     state['price'] = normalize(state['price'].reshape(1, -1))[0]
    #     state['holding'] = normalize(state['holding'].reshape(1, -1))[0]
    #     state['balance'] = state['balance'] / self.start_balance
    #     return state

class Actor(nn.Module):
    def __init__(self,input_size,output_size):
        super(Actor,self).__init__()
        self.linear1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE1)
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2)
        self.linear3 = nn.Linear(HIDDEN_LAYER_SIZE2, output_size)
        # self.linear1.weight.data.normal_(0, 0.001)
        # self.linear2.weight.data.normal_(0, 0.001)
        # self.linear3.weight.data.normal_(0, 0.001)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
    def forward(self, state):
        h1 = self.linear1(state)
        h2 = self.linear2(self.relu(h1))
        h2 = self.relu(h2)
        out = self.tanh(self.linear3(h2))
        return out * 3

class Critic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Critic,self).__init__()
        self.linear1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE1)
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2)
        self.linear3 = nn.Linear(HIDDEN_LAYER_SIZE2, output_size)
        # self.linear1.weight.data.normal_(0, 0.001)
        # self.linear2.weight.data.normal_(0, 0.001)
        # self.linear3.weight.data.normal_(0, 0.001)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
    def forward(self, state, action):
        x = torch.cat([state,action],1)
        h1 = self.linear1(x)
        h2 = self.linear2(self.relu(h1))
        h2 = self.relu(h2)
        out = self.linear3(h2)
        return out 