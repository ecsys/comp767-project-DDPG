import math
import numpy as np
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding


# from scipy.special import softmax

class StockEnv(gym.Env):

    def __init__(self, start_date='2009-01-02', end_date='2016-01-04', start_balance=10000, transaction_fee=0,
                 threshold=0):

        # constant
        tickers = ["AXP", "AAPL", "BA", "CAT", "CSCO",
                   "CVX", "DWDP", "XOM", "GS", "HD",
                   "IBM", "INTC", "JNJ", "KO", "JPM",
                   "MCD", "MMM", "MRK", "MSFT", "NKE",
                   "PFE", "PG", "TRV", "UNH", "UTX",
                   "VZ", "WBA", "WMT", "DIS"]

        data_dir = '../data/'

        self.start_date = start_date
        self.end_date = end_date
        self.n_stock = len(tickers)
        self.tickers = tickers
        self.start_balance = start_balance
        self.data_dir = data_dir
        self.transaction_fee = transaction_fee
        self.threshold = threshold

        # read all data into memory when initializing
        self.stock_data = self.load_data(data_dir, tickers)

        # initialize state, action, and date indices
        self.state = {'price': np.zeros(self.n_stock),
                      'holding': np.zeros(self.n_stock, dtype=np.int64),
                    #   'volume': np.zeros(self.n_stock),
                      'balance': 0}

        self.action = np.zeros(self.n_stock)  # selling quantity
        self.action_space = np.stack([np.array([-5, 5]) for i in range(self.n_stock)])
        self.state_space_size = self.n_stock * 2 + 1

        self.date_pointer = []
        self.done = False
        self.reset()

    def step(self, action):
        
        self.action = np.round(action)

        curr_total = self.get_market_value(self.state)
        next_state = self.load_next_day_state(action)
        next_total = self.get_market_value(next_state)
        reward = next_total - curr_total

        # move one day forward
        self.date_pointer = [date - 1 for date in self.date_pointer]

        # done if passed end_date
        date = self.get_date_from_index(self.date_pointer[0])
        if date >= self.end_date:
            self.done = True

        # calculate transaction fee
        for a in action:
            if a != 0:
                reward -= self.transaction_fee

        return self.state, reward, self.done, action

    def reset(self):

        self.done = False
        self.date_pointer = self.get_index_from_date(self.start_date)

        prices = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = self.date_pointer[idx]
            prices.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
        self.state['price'] = np.array(prices)
        self.state['holding'] = np.zeros(self.n_stock, dtype=np.int64)
        self.state['balance'] = self.start_balance
        # self.state['volume'] = np.zeros(self.n_stock, dtype=np.int64)
        self.action = np.zeros(self.n_stock)
        self.action_space = np.stack([np.array([-5, 5]) for i in range(self.n_stock)])

        return self.state

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

        for idx, action_ in enumerate(action):
            max_buy = -np.floor(self.state['balance'] / self.state['price'][idx])
            max_sell = self.state['holding'][idx]
            if action_ > 0:
                action_ = min(action_, max_sell)
            else:
                action_ = max(action_, max_buy)

            # execute action
            self.state['holding'][idx] -= action_
            self.state['balance'] += action_ * self.state['price'][idx]

        # then advances price and volume to the next state
        date_pointer = list(self.date_pointer)
        next_date_pointer = [date - 1 for date in date_pointer]
        volume = []
        next_price = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = next_date_pointer[idx]
            next_price.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
            # volume.append(self.stock_data[ticker].iloc[ticker_row_idx]['volume'])
        next_price = np.array(next_price)
        # volume = np.array(volume)
        self.state['price'] = next_price
        # self.state['volume'] = volume

        return self.state

    def get_market_value(self, state):
        market_value = state['holding'].dot(state['price'])
        total = market_value + state['balance']
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

    def is_valid_action(self, action):
        valid_action = 0
        amount_required = 0
        amount_gain = 0
        for i in range(len(self.tickers)):
            # cannot sell or buy partial stock
            if not isinstance(action[i], np.int64):
                valid_action = 1
                # print("invalid action 1")
                break
            # cannot sell more than you have, no short operation allowed
            if self.state['holding'][i] < action[i]:
                valid_action = 2
                # print("invalid action 2")
                break
            if action[i] < 0:
                amount_required += abs(action[i]) * self.state['price'][i]
            if action[i] > 0:
                amount_gain += abs(action[i]) * self.state['price'][i]

        # cannot spend more money than you have to buy stocks
        if amount_required > self.state['balance'] + amount_gain:
            valid_action = 3
            # print("invalid action 3")

        return valid_action

    def clip_action(self, action):

        prices = self.state['price']
        holdings = self.state['holding']
        balance = self.state['balance']

        for i in range(len(action)):
            if action[i] > holdings[i]:
                action[i] = holdings[i]
        total = -1 * np.sum(action * prices)
        while total > balance:
            for i in range(len(action)):
                if action[i] < 0:
                    action[i] *= 0.8
            total = -1 * np.sum(action * prices)
        return action.astype(np.int)
