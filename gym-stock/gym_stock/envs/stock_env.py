import math
import numpy as np
import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class StockEnv(gym.Env):

    def __init__(self, tickers, train_start_date, train_end_date, start_balance):

        # constant
        tickers = ["AXP","AAPL","BA","CAT","CSCO",
                "CVX","DWDP","XOM","GS","HD",
                "IBM","INTC","JNJ","KO","JPM",
                "MCD","MMM","MRK","MSFT","NKE",
                "PFE","PG","TRV","UNH","UTX",
                "VZ","V","WBA","WMT","DIS"]
        
        data_dir = ''

        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.n_stock = len(tickers)
        self.tickers = tickers
        self.start_balance = start_balance
        self.data_dir =data_dir

        # read all data into memory when initializing
        self.stock_data = self.load_data(data_dir, tickers)
        
        # initialize state, action, and date indices
        self.state = {'price': np.zeros(self.n_stock),
                      'holding': np.zeros(self.n_stock),
                      'balance': 0}

        self.action = np.zeros(self.n_stock) # selling quantity
        self.action_space = np.zeros((self.n_stock,2))
        self.state_space = np.zeros((self.n_stock,2))
        #TODO state space: for price holding balance -inf to inf??
        self.date_pointer = []
        self.done = False
        self.reset()

    def step(self, action):
        assert self.is_valid_action(action)

        self.action = action
        curr_total = self.get_market_value(self.state)
        next_state = self.load_next_day_state(action)
        next_total = self.get_market_value(next_state)
        reward = next_total - curr_total

        self.action_space = self.get_action_space(next_state)
        self.state = next_state

        # move one day forward
        self.date_pointer = [date + 1 for date in self.date_pointer]

        # done if passed train_end_date
        date = self.get_date_from_index(self.date_pointer[0])
        if date >= self.train_end_date:
            self.done = True
        
        return self.state, reward, self.done 

    def reset(self):
        
        self.done = False
        self.date_pointer = self.get_index_from_date(self.train_start_date)

        prices = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = self.date_pointer[idx]
            prices.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
        self.state['price'] = np.array(prices)
        self.state['holding'] = np.zeros(self.n_stock)
        self.state['balance'] = self.start_balance

        self.action = np.zeros(self.n_stock)
        self.action_space = np.zeros((self.n_stock,2))
        return self.state

    def render(self):
        pass

    def get_action_space(self, state):
        action_space = []
        prices = state['price']
        holdings = state['holding']
        balance = state['balance']
        for idx, price in enumerate(prices):
            max_buy = math.floor(balance / price)
            max_sell = holdings[idx]
            action_space.append([-max_buy, max_sell])
        return np.array(action_space)

    def load_data(self, data_dir, stocks):
        stock_data = {}
        for ticker in stocks:
            file_name = data_dir + ticker + '.csv'
            data = pd.read_csv(file_name)
            stock_data[ticker] = data
        return stock_data

    def load_next_day_state(self, action):
        assert self.is_valid_action(action)

        date_pointer = self.date_pointer.copy()
        next_date_pointer = [date + 1 for date in date_pointer]

        next_price = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = next_date_pointer[idx]
            next_price.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
        next_price = np.array(next_price)

        next_holding = self.state['holding'] - action

        next_balance = self.state['balance']
        for idx, action_amt in enumerate(action):
            next_balance += self.state['price'][idx] * action_amt

        next_state = {'price': next_price,
                      'holding': next_holding,
                      'balance': next_balance}
        return next_state        

    def get_market_value(self, state):
        market_value = 0
        for idx, holding_amt in enumerate(state['holding']):
            market_value += holding_amt * state['price'][idx]
        total = market_value + state['balance']
        return total

    def get_index_from_date(self, date):
        stock_date_index = []
        for ticker in self.tickers:
            stock_df = self.stock_data[ticker]
            stock_date_index.append(stock_df[stock_df['timestamp']==date].index.values.astype(int)[0])
        return stock_date_index

    def get_date_from_index(self, index):
        stock_df = self.stock_data[self.tickers[0]]
        date = stock_df.iloc[index]['timestamp']
        return date

    def is_valid_action(self, action):
        valid_action = True
        amount_required = 0
        amount_gain = 0
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
            if action[i] > 0:
                amount_gain += abs(action[i]) * self.state['price'][i]

        #cannot spend more money than you have to buy stocks
        if amount_required > self.state['balance'] + amount_gain:
            valid_action = False

        return valid_action
        