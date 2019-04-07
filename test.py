from DDPG import DDPG
from generateNoise import OUNoise
from hyperparameters import *
from stock_env import *
env = StockEnv()
rewards = []
agent = DDPG(env)
noise = OUNoise(env.action_space)
for eposide in range(100):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.sample_action(state)
        action = noise.get_action(action=action)
        new_state, reward, done = env.step(action)
        agent.memory.push(state,action,reward,new_state,done)
        if agent.memory.check_full():
            agent.update(BATCH_SIZE)
        state = new_state
        episode_reward+= reward
        if done:
            print('eposide: {}, reward: {}'.format(eposide,reward))
            break
    rewards.append(episode_reward)




