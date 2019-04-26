import random
from collections import deque
from copy import deepcopy

class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self, size):
        assert size < self.capacity
        batch = random.sample(self.buffer, size)
        action = []
        state = []
        reward = []
        next_state = []
        done = []
        for exp in batch:
            action.append(exp.action)
            state.append(exp.state)
            reward.append(exp.reward)
            next_state.append(exp.next_state)
            done.append(exp.done)
        return state, action, reward, next_state, done

    def check_full(self):
        return len(self.buffer) == self.capacity
