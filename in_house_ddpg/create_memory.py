from stock_env import *
import numpy as np
env = StockEnv()



def create_action(state):
    price = state['price']
    holding = state['holding']
    balance = state['balance']
    action = np.random.uniform(-1,1,29)
    # for h in range(len(holding)):
    #     action[h] = np.random.uniform(0,100)
    return action


def create_memory(agent):
    for eposide in range(3):
       state = env.reset()
       episode_reward = 0
       done = False
       step_num = 0
       while not done:
           step_num += 1
           action = create_action(state)
           action = np.array(action, dtype=np.int64)
           new_state, reward, done, clipped_action = env.step(action)
           agent.memory.push(state, clipped_action, reward, new_state, done)
           state = new_state
           episode_reward += reward
           if step_num % 500 == 0:
               print('filling up memory eposide: {}, step: {}, reward: {}'.format(eposide, step_num, episode_reward))
           if done:
               print('eposide: {}, reward: {}'.format(eposide, episode_reward))
               break