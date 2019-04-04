import numpy as np
import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class StockEnv(gym.Env):

    def __init__(self, tickers, train_start_date, start_balance):


        # constant
        tickers = ["AXP","AAPL","BA","CAT","CSCO",
                "CVX","DWDP","XOM","GS","HD",
                "IBM","INTC","JNJ","KO","JPM",
                "MCD","MMM","MRK","MSFT","NKE",
                "PFE","PG","TRV","UNH","UTX",
                "VZ","V","WBA","WMT","DIS"]
        
        data_dir = ''

        self.train_start_date = train_start_date
        self.n_stock = len(tickers)
        self.tickers = tickers
        self.start_balance = start_balance

        # read all data into memory when initializing
        self.stock_data = self.load_data(data_dir, tickers)
        
        # initialize state, action, and date indices
        self.state = {'price': np.zeros(self.n_stock),
                      'holding': np.zeros(self.n_stock),
                      'balance': 0}
        self.action = np.zeros(self.n_stock)
        self.current_day = [] 
        self.reset()

    def step(self, action):
        pass
        # def get_reward(self, cur_state, action, next_state):

    def reset(self):
        
        self.current_day = self.get_index_from_date(self.train_start_date)

        prices = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = self.current_day[idx]
            prices.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
        self.state['price'] = np.array(prices)
        self.state['holding'] = np.zeros(self.n_stock)
        self.state['balance'] = self.start_balance

        self.action = np.zeros(self.n_stock)

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

    def get_index_from_date(self, date):
        stock_date_index = []
        for ticker in self.tickers:
            stock_df = self.stock_data[ticker]
            stock_date_index.append(stock_df[stock_df['timestamp']==date].index.values.astype(int)[0])
        return stock_date_index

    def is_valid_action(self, action):
        valid_action = True
        amount_required = 0
        for i in range(len(self.tickers)):
            # cannot sell or buy partial stock
            if not isinstance(action[i], np.int64):
                valid_action = False
                break
            # cannot sell more than you have, no short operation allowed
            if not self.state['holding'][i] >= action[i]:
                valid_action = False
                break
            if action[i] < 0:
                amount_required += abs(action[i]) * self.state['price'][i]
        # cannot spend more money than you have to buy stocks
        if amount_required > self.state['balance']:
            valid_action = False

        return valid_action
        
        

        

    
    


