import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start_date='2010-01-05', end_date='2015-01-02', 
                 start_balance=10000, transaction_fee=0, threshold=0, scale=5):

        # constant
        tickers = ["AXP", "AAPL", "BA", "CAT", "CSCO",
                   "CVX", "DWDP", "XOM", "GS", "HD",
                   "IBM", "INTC", "JNJ", "KO", "JPM",
                   "MCD", "MMM", "MRK", "MSFT", "NKE",
                   "PFE", "PG", "TRV", "UNH", "UTX",
                   "VZ", "WBA", "WMT"]

        data_dir = '/home/ecsys/Dropbox/UdeM/COMP767/comp767-final/data/'

        self.tickers = tickers
        self.n_stock = len(self.tickers)
        self.scale = scale
        
        self.action_space = spaces.Box(low = -self.scale, high = self.scale,shape = (self.n_stock,),dtype=np.int8) 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.n_stock*2+1,))

        self.start_date = start_date
        self.end_date = end_date
        
        self.start_balance = start_balance
        self.data_dir = data_dir
        self.transaction_fee = transaction_fee
        self.threshold = threshold

        # read all data into memory when initializing
        self.stock_data = self.load_data(data_dir, self.tickers)
        self.terminal = False
        
        # initialize state, action, and date indices
        self.state = [self.start_balance] + [0 for i in range(2 * self.n_stock)]

        self.action = np.zeros(self.n_stock)  # selling quantity
        self.reward = 0
        self.date_pointer = []

        self.asset_list = [self.start_balance]
        self.iteration = 0

        self.reset()
        self._seed()

        
        
    def step(self, actions):

        # done if passed end_date
        date = self.get_date_from_index(self.date_pointer[0])
        if date >= self.end_date:
            self.terminal = True

        if self.terminal:
            plt.plot(self.asset_list,'r')
            plt.savefig('/home/ecsys/iteration_{}.png'.format(self.iteration))
            plt.close()
            print("total_reward:{}".format(self.get_market_value(self.state)- 10000))
            self.iteration += 1

            return self.state, self.reward, self.terminal,{}

        else:
            # actions = np.round(actions)

            curr_total = self.get_market_value(self.state)
            next_state = self.load_next_day_state(actions)
            next_total = self.get_market_value(next_state)
            self.reward = next_total - curr_total

            # move one day forward
            self.date_pointer = [date - 1 for date in self.date_pointer]

            # calculate transaction fee
            for a in actions:
                if a != 0:
                    self.reward -= self.transaction_fee

            self.asset_list.append(next_total)

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.terminal = False
        self.date_pointer = self.get_index_from_date(self.start_date)
        self.reward = 0

        prices = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = self.date_pointer[idx]
            prices.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
        self.state = [self.start_balance] + prices + [0 for i in range(self.n_stock)]
        self.action = [0 for i in range(self.n_stock)]
        self.action_space = spaces.Box(low = -self.scale, high = self.scale,shape = (self.n_stock,),dtype=np.int8) 

        self.asset_list = [self.start_balance]

        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def load_data(self, data_dir, stocks):
        stock_data = {}
        for ticker in stocks:
            file_name = data_dir + ticker + '.csv'
            data = pd.read_csv(file_name)
            stock_data[ticker] = data
        return stock_data

    def load_next_day_state(self, action):

        # first sell
        for idx, action_ in enumerate(action):
            max_sell = self.state[1+self.n_stock+idx]
            if action_ > 0:
                action_ = min(action_, max_sell)
                self.state[1+self.n_stock+idx] -= action_
                self.state[0] += action_ * self.state[1+idx]

        # buy
        buy_action = [min(0, a) for a in action]
        argsort_buy = list(np.argsort(buy_action))
        for i in range(self.n_stock):
            action_idx = argsort_buy.index(i)
            max_buy = -np.floor(self.state[0] / self.state[action_idx+1])
            action_ = max(buy_action[action_idx], max_buy)
            self.state[1+self.n_stock+idx] -= action_
            self.state[0] += action_ * self.state[1+idx]

        # then advances price and volume to the next state
        date_pointer = list(self.date_pointer)
        next_date_pointer = [date - 1 for date in date_pointer]
        volume = []
        next_price = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = next_date_pointer[idx]
            next_price.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
        self.state[1:1+self.n_stock] = next_price

        return self.state

    def get_market_value(self, state):
        market_value = np.array(state[1:self.n_stock+1]).dot(np.array(state[self.n_stock+1:]))
        total = market_value + state[0]
        return total

    def get_index_from_date(self, date):
        stock_date_index = []
        for ticker in self.tickers:
            stock_df = self.stock_data[ticker]
            stock_date_index.append(stock_df[stock_df['timestamp'] == date].index.values.astype(int)[0])
        return stock_date_index

    def get_date_from_index(self, index):
        stock_df = self.stock_data[self.tickers[0]]
        date = stock_df.iloc[index]['timestamp']
        return date