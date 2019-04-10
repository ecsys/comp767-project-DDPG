import math
import numpy as np
import pandas as pd
import gym
from gym import error, spaces, utils
from gym.utils import seeding


# from scipy.special import softmax

class StockEnv(gym.Env):

    def __init__(self, start_date='2000-01-03', end_date='2015-01-02', start_balance=100000, transaction_fee=0,
                 threshold=0):

        # constant
        tickers = ["AXP", "AAPL", "BA", "CAT", "CSCO",
                   "CVX", "DWDP", "XOM", "GS", "HD",
                   "IBM", "INTC", "JNJ", "KO", "JPM",
                   "MCD", "MMM", "MRK", "MSFT", "NKE",
                   "PFE", "PG", "TRV", "UNH", "UTX",
                   "VZ", "WBA", "WMT", "DIS"]

        data_dir = 'data/'

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
                      'volume': np.zeros(self.n_stock),
                      'balance': 0}

        self.action = np.zeros(self.n_stock)  # selling quantity
        self.action_space = np.zeros((self.n_stock, 2))
        self.state_space_size = self.n_stock * 3 + 1
        # self.state_space = np.zeros((self.n_stock,3))
        # TODO state space: for price holding balance -inf to inf??
        self.date_pointer = []
        self.done = False
        self.reset()

    def step(self, action):
        # if self.is_valid_action(action) != 0:
        #     action = self.clip_action(action)

        curr_total = self.get_market_value(self.state)
        next_state, action = self.load_next_day_state(action)
        self.action = action
        next_total = self.get_market_value(next_state)
        reward = next_total - curr_total

        self.action_space = self.get_action_space(next_state)

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
        self.state['volume'] = np.zeros(self.n_stock, dtype=np.int64)
        self.action = np.zeros(self.n_stock)
        self.action_space = self.get_action_space(self.state)
        return self.state

    def render(self):
        pass

    def get_action_space(self, state):
        action_space = []
        prices = state['price']
        holdings = state['holding']
        balance = state['balance']
        for idx, price in enumerate(prices):
            # max_buy = math.floor(balance / price)
            max_buy = 100
            max_sell = holdings[idx]
            action_space.append([-max_buy, max_sell])
        return np.array(action_space, dtype=np.int64)

    def load_data(self, data_dir, stocks):
        stock_data = {}
        for ticker in stocks:
            file_name = data_dir + ticker + '.csv'
            data = pd.read_csv(file_name)
            stock_data[ticker] = data
        return stock_data

    def load_next_day_state(self, action):
        # first trade do the action,
        action = self.clip_action(action)
        next_holding = self.state['holding'] - action
        next_balance = self.state['balance']
        next_balance += np.sum(self.state['price'] * action)
        date_pointer = list(self.date_pointer)
        self.state['holding'] = next_holding
        self.state['balance'] = next_balance

        # then advances to the next state
        next_date_pointer = [date - 1 for date in date_pointer]
        volume = []
        next_price = []
        for idx, ticker in enumerate(self.tickers):
            ticker_row_idx = next_date_pointer[idx]
            next_price.append(self.stock_data[ticker].iloc[ticker_row_idx]['open'])
            volume.append(self.stock_data[ticker].iloc[ticker_row_idx]['volume'])
        next_price = np.array(next_price)
        volume = np.array(volume)
        self.state['price'] = next_price
        self.state['volume'] = volume

        return self.state, action

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
        # validity_code = self.is_valid_action(action)
        # for i in range(len(action)):
        #     if action[i] >= holdings[i]:
        #         action[i] = holdings[i]
        # while (validity_code != 0):
        #     if validity_code == 3:
        #         action = np.where(action < 0, action * 0.8, action)
        #         action = action.astype(np.int64)
        #     elif validity_code == 2:
        #         for i in range(len(self.tickers)):
        #             if holdings[i] < action[i]:
        #                 action[i] = holdings[i]
        #     validity_code = self.is_valid_action(action)

        for i in range(len(action)):
            while action[i] > holdings[i]:
                action[i] *= 0.8
        total = -1 * np.sum(action * prices)
        while total > balance:
            for i in range(len(action)):
                if action[i] < 0:
                    action[i] *= 0.8
            total = -1 * np.sum(action * prices)
        return action.astype(np.int)
        # # cut small amount transactions
        # for idx, _action in enumerate(action):
        #     if _action == holdings[idx]:
        #         continue
        # 
        # validity_code = self.is_valid_action(action)
        # if  validity_code== 0:
        #     return action
        # 
        # while (validity_code != 0):
        #     if validity_code == 3:
        #         action = np.where(action<0, action*0.9, action)
        #         action = action.astype(np.int64)
        #     elif validity_code == 2:
        #         for i in range(len(self.tickers)):
        #             if holdings[i] < action[i]:
        #                 action[i] = holdings[i]
        #     validity_code = self.is_valid_action(action)
        # 
        # amount_delta = 0
        # for idx, _action in enumerate(action):
        #     amount_delta += _action * prices[idx]
        # alpha = 1
        # while(alpha * amount_delta + balance < 0):
        #     alpha -= 0.01
        # action = (alpha * action).astype(np.int64)

        # buy_action = np.absolute(np.where(action<0, action, 0))
        # if sum(buy_action) != 0:
        #     buy_action = softmax(np.where(buy_action!=0, action, -1e10)) * -1

        # sell_action = np.where(action>0, action, 0)
        # if sum(sell_action) != 0:
        #     sell_action = softmax(np.where(sell_action!=0, action, -1e10))
        # action = buy_action + sell_action

        # normalized_amount = np.absolute(action.T.dot(prices))
        # alpha = balance/normalized_amount
        # action = (action * alpha).astype(np.int64)

        # return action
