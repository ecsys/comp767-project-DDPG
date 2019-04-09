from DDPG import DDPG
from generateNoise import OUNoise
from hyperparameters import *
from stock_env import *

def test_actor(start_date, end_date, agent):
    test_env = StockEnv(start_date=start_date, end_date=end_date)
    total_reward = 0
    done = False
    while not done:
        action = agent.sample_action(test_env.state)
        action = np.array(action, dtype=np.int64)
        _, reward, done, action = env.step(action)
        total_reward += reward
    print('Test: start date {}, end-date {}, reward: {}'.format(start_date, end_date, total_reward))
    return total_reward

env = StockEnv(end_date='2017-04-03')
rewards = []
agent = DDPG(env)
noise = OUNoise(env.action_space)
test_best_reward = -1e10
for eposide in range(100):
    state = env.reset()
    episode_reward = 0
    done = False
    step_num = 0
    while not done:
        step_num += 1
        noise.set_action_space(env.action_space)
        action = agent.sample_action(state)
        action = noise.get_action(action=action, t=step_num)
        action = np.array(action, dtype=np.int64)
        new_state, reward, done, clipped_action = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)
        if agent.memory.check_full():
            agent.update(BATCH_SIZE)
        state = new_state
        episode_reward += reward
        if step_num % 500 == 0:
            print('eposide: {}, step: {}, reward: {}'.format(eposide, step_num, reward))
        if done:
            print('eposide: {}, reward: {}'.format(eposide, episode_reward))
            break
    test_reward = test_actor('2017-04-03', '2019-04-01', agent)
    rewards.append(episode_reward)
    if test_reward > test_best_reward:
        test_best_reward = test_reward
        agent.save()
        print("model saved with test_reward {}".format(test_reward))
