from copy import deepcopy
import torch.autograd
from generateNoise import OUNoise
from hyperparameters import *
from memory import *
from models import *
import torch.optim as optim
import torch.nn as nn
class DDPG:
    def __init__(self, env, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, tau=TAU, discount_rate=DISCOUNT_RATE, memory_size=MEMORY_SIZE):
        self.state_num = env.state_space_size
        self.action_num = env.action_space.shape[0]
        self.gamma = discount_rate
        self.tau = tau
        self.actor_net = Actor(self.state_num, self.action_num)
        self.critic_net = Critic(self.state_num + self.action_num, 1)
        self.target_actor = deepcopy(self.actor_net)
        self.target_critic = deepcopy(self.critic_net)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.critic_loss = nn.MSELoss()
        self.noise = OUNoise(env.action_space, mu=np.zeros(self.action_num))
        self.memory = Memory(memory_size)

    def sample_action(self, state):
        self.actor_net.eval()
        state = np.concatenate([state['price'],state['holding'],[state['balance']]])
        state = torch.from_numpy(state).float()

        state[:29] = state[:29] / max(state[:29])
        state[29:-1] = state[29:-1] / (max(state[29:-1]) + 1e-10)
        state[-1] = state[-1] / 10000
        state = state.unsqueeze(0)

        action = self.actor_net.forward(state)
        action = action.data.numpy()[0]
        return action

    def normalize_state(self, state):
        state[:, :29] = state[:, :29] / np.max(state[:, :29], axis=1).reshape(-1, 1)
        state[:, 29:-1] = state[:, 29:-1] / (np.max(state[:, 29:-1], axis=1).reshape(-1, 1) + 1e-10)
        state[:, -1] = state[:, -1] / 10000
        return state

    def update(self, batch_size=BATCH_SIZE):
        self.actor_net.train()
        self.critic_net.train()

        state_list, action_list, reward_list, next_state_list, done_list = self.memory.sample(batch_size)

        states = []
        for state in state_list:
            s = np.concatenate([state['price'], state['holding'], [state['balance']]])
            states.append(s)
        state_list = np.array(states)
        next_states = []
        for next_state in next_state_list:
            s = np.concatenate([next_state['price'], next_state['holding'], [next_state['balance']]])
            next_states.append(s)
        next_state_list = np.array(next_states)

        state_list = self.normalize_state(state_list)
        next_state_list = self.normalize_state(next_state_list)
        
        action_list = torch.FloatTensor(action_list)
        reward_list = torch.FloatTensor(reward_list)
        state_list = torch.FloatTensor(state_list)
        next_state_list = torch.FloatTensor(next_state_list)

        Qvals = self.critic_net(state_list, action_list)
        next_actions = self.target_actor(next_state_list)

        next_Qvals = self.target_critic(next_state_list, next_actions)
        y = reward_list + self.gamma * next_Qvals

        self.critic_optim.zero_grad()
        critic_loss = self.critic_loss(Qvals, y)
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = -torch.mean(self.critic_net(state_list,self.actor_net(state_list)))
        policy_loss.backward()
        self.actor_optim.step()


        for target_param, param in zip(self.target_actor.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
			
    def save(self):
        torch.save({
            'model_state_dict': self.actor_net.state_dict(),
            'optimizer_state_dict': self.actor_optim.state_dict()

        }, ACTOR_PATH)
        torch.save({
            'model_state_dict': self.critic_net.state_dict(),
            'optimizer_state_dict': self.critic_net.state_dict(),
            'loss': self.critic_loss
        }, CRITIC_PATH)
        torch.save({
            'model_state_dict': self.target_actor.state_dict(),

        }, ACTOR_TARGET_PATH)
        torch.save({
            'model_state_dict': self.target_critic.state_dict(),
        }, CRITIC_TARGET_PATH)
    def load(self):

        actor_checkpoint = torch.load(ACTOR_PATH)
        self.actor_net.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optim.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        critic_checkpoint = torch.load(CRITIC_PATH)
        self.critic_net.load_state_dict(critic_checkpoint['model_state_dict'])
        self.critic_net.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        self.critic_loss = critic_checkpoint['loss']

        actor_target_checkpoint = torch.load(ACTOR_TARGET_PATH)
        self.target_actor.load_state_dict(actor_target_checkpoint['model_state_dict'])

        critic_target_checkpoint = torch.load(CRITIC_TARGET_PATH)
        self.target_critic.load_state_dict(critic_target_checkpoint['model_state_dict'])
