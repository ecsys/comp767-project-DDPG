from copy import deepcopy
from torch.autograd import Variable
from generateNoise import OUNoise
from hyperparameters import *
from memory import *
from models import *


class DDPG:
    def __init__(self, env, actor_lr=1e-4, critic_lr=1e-4, tau=TAU, discount_rate=0.99, memory_size=MEMORY_SIZE):
        self.state_num = env.state_space_size
        self.action_num = env.action_space_size
        self.gamma = discount_rate
        self.tau = tau
        self.actor_net = Actor(self.state_num, self.action_num)
        self.critic_net = Critic(self.state_num + self.action_num, self.action_num)
        self.target_actor = deepcopy(self.actor_net)
        self.target_critic = deepcopy(self.critic_net)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.critic_loss = nn.MSELoss()
        self.noise = OUNoise(env.action_space, mu=np.zeros(self.action_num))
        self.memory = Memory(memory_size)

    def sample_action(self, state):
        state = np.concatenate([state['price'],state['holding'],state['balance']])
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor_net.forward(state)
        action = action.detach().numpy()[0, 0]
        action = self.noise.get_action(action)
        return action

    def update(self, batch_size=BATCH_SIZE):
        state_list, action_list, reward_list, next_state_list, done_list = self.memory.sample(batch_size)
        action_list = torch.FloatTensor(action_list)
        reward_list = torch.FloatTensor(reward_list)
        state_list = torch.FloatTensor(state_list)
        next_state_list = torch.FloatTensor(next_state_list)
        Qvals = self.critic_net.forward(state_list,action_list)
        next_actions = self.target_actor.forward(next_state_list)
        next_Qvals = self.target_critic.forward(next_state_list,next_actions)
        y = reward_list+self.gamma*next_Qvals
        critic_loss = self.critic_loss(Qvals,y)
        policy_loss = -torch.mean(self.critic_net.forward(state_list,self.actor_net.forward(state_list)))

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        for target_param, param in zip(self.target_actor.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_critic.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))