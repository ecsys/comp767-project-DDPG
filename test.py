from DDPG import DDPG
from generateNoise import OUNoise
from hyperparameters import *
from stock_env import *

def trade_one_stock(action):
    sell = max(action)
    buy = min(action)
    if abs(sell) > abs(buy):
        argidx = np.argmax(action)
    else:
        argidx = np.argmin(action)
    for i in range(len(action)):
        if i != argidx:
            action[i] = 0
    return action

env = StockEnv()
rewards = []
agent = DDPG(env)
noise = OUNoise(env.action_space)
for eposide in range(100):
    state = env.reset()
    episode_reward = 0
    done = False
    step_num = 0
    while not done:
        step_num += 1
        noise.set_action_space(env.action_space)
        
        action = agent.sample_action(state)
        action = noise.get_action(action=action)
        action = trade_one_stock(np.array([int(a) for a in action]))
        new_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)
        if agent.memory.check_full():
            agent.update(BATCH_SIZE)
        state = new_state
        episode_reward += reward
        if step_num % 500 == 0:
            print('eposide: {}, step: {}, reward: {}'.format(eposide, step_num, reward))
        if done:
            print('eposide: {}, reward: {}'.format(eposide, reward))
            break
    rewards.append(episode_reward)