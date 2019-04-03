import numpy as np
import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class StockEnv(gym.Env):

    # constant
    DOW_JONES = ["AXP","AAPL","BA","CAT","CSCO",
             "CVX","DWDP","XOM","GS","HD",
             "IBM","INTC","JNJ","KO","JPM",
             "MCD","MMM","MRK","MSFT","NKE",
             "PFE","PG","TRV","UNH","UTX",
             "VZ","V","WBA","WMT","DIS"]

    def __init__(self, n_stock, train_start, train_end, test_start, test_end):
        
        self.state = {'price': np.zeros(n_stock),
                      'holding': np.zeros(n_stock),
                      'balance': 0}

        self.action = np.zeros(n_stock)

        self.current_day = {}

    def step(self, action):
        pass
        # def get_reward(self, cur_state, action, next_state):

    def reset(self):
        pass

    def render(self):
        pass

    def load_data(self, data_dir, stocks):
        stock_data = {}
        for ticker in stocks:
            file_name = data_dir + ticker + '.csv'
            data = pd.read_csv(file_name)
            stock_data[ticker] = data
        return stock_data

    def load_next_day_state(self):
        pass
    
    


