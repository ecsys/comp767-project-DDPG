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
        self.action = np.zeros(self.n_stock) # selling quantity
        self.date_pointer = [] 
        self.reset()

    def step(self, action):
        assert self.is_valid_action(action)

        next_state = self.load_next_day_state(action)
        next_total = self.get_market_value(next_state)

        # move one day forward
        self.date_pointer = [date + 1 for date in self.date_pointer]
        
        return next_total # is reward the portfolio value?

    def reset(self):
        
        self.date_pointer = self.get_index_from_date(self.train_start_date)

        prices = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = self.date_pointer[idx]
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

    def load_next_day_state(self, action):
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
        