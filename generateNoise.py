import numpy as np
from hyperparameters import *
"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=SIGMA, min_sigma=0, decay_period=3000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space[:,0]
        self.high = action_space[:,1]
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * max(min(1.0, t / self.decay_period),0)
        return np.clip(action + ou_state, self.low, self.high)
    
    def set_action_space(self, action_space):
        self.low = action_space[:,0]
        self.high = action_space[:,1]