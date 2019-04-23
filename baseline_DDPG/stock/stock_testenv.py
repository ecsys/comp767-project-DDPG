import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt

# dji = pd.read_csv("/home/ecsys/anaconda3/envs/tensorflow/lib/python3.6/site-packages/gym/envs/zxstock/Data_Daily_Stock_Dow_Jones_30/^DJI.csv")
# test_dji=dji[dji['Date']>'2016-01-01']
# dji_price=test_dji['Adj Close']
# dji_date = test_dji['Date']
# daily_return = dji_price.pct_change(1)
# daily_return=daily_return[1:]
# daily_return.reset_index()
# initial_amount = 10000

# total_amount=initial_amount
# account_growth=list()
# account_growth.append(initial_amount)
# for i in range(len(daily_return)):
#     total_amount = total_amount * daily_return.iloc[i] + total_amount
#     account_growth.append(total_amount)



# data_1 = pd.read_csv('/home/ecsys/anaconda3/envs/tensorflow/lib/python3.6/site-packages/gym/envs/zxstock/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv')

# equal_4711_list = list(data_1.tic.value_counts() == 4711)
# names = data_1.tic.value_counts().index

# # select_stocks_list = ['NKE','KO']
# select_stocks_list = list(names[equal_4711_list])+['NKE','KO']

# data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912','20010913'])]

# data_3 = data_2[['iid','datadate','tic','prccd','ajexdi']]

# data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

# # train_data = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
# test_data = data_3[data_3.datadate > 20160000]

# # train_daily_data = []

# # for date in np.unique(train_data.datadate):
# #     train_daily_data.append(train_data[train_data.datadate == date])

# # print(len(train_daily_data)) 
# test_daily_data = []

# for date in np.unique(test_data.datadate):
#     test_daily_data.append(test_data[test_data.datadate == date])

# # whole_data = train_daily_data+test_daily_data

# iteration = 0


# class StockTestEnv(gym.Env):
#     metadata = {'render.modes': ['human']}

#     def __init__(self, day = 0, money = 10 , scope = 1):
#         self.day = day
#         # self.money = money
        
#         # buy or sell maximum 5 shares
#         self.action_space = spaces.Box(low = -5, high = 5,shape = (28,),dtype=np.int8) 

#         # # buy or sell maximum 5 shares
#         # self.action_space = spaces.Box(low = -5, high = 5,shape = (2,),dtype=np.int8) 

#         # [money]+[prices 1-28]+[owned shares 1-28]
#         self.observation_space = spaces.Box(low=0, high=np.inf, shape = (57,))

#         # # [money]+[prices 1-28]+[owned shares 1-28]
#         # self.observation_space = spaces.Box(low=0, high=np.inf, shape = (5,))
        
#         self.data = test_daily_data[self.day]
        
#         self.terminal = False
        
#         self.state = [10000] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
#         self.reward = 0
        
#         self.asset_memory = [10000]

#         self.reset()
#         self._seed()


#     def _sell_stock(self, index, action):
#         if self.state[index+29] > 0:
#             self.state[0] += self.state[index+1]*min(abs(action), self.state[index+29])
#             self.state[index+29] -= min(abs(action), self.state[index+29])
#         else:
#             pass
    
#     def _buy_stock(self, index, action):
#         available_amount = self.state[0] // self.state[index+1]
#         # print('available_amount:{}'.format(available_amount))
#         self.state[0] -= self.state[index+1]*min(available_amount, action)
#         # print(min(available_amount, action))

#         self.state[index+29] += min(available_amount, action)
        
#     def step(self, actions):
#         # print(self.day)
#         self.terminal = self.day >= 685
#         # print(actions)

#         if self.terminal:
#             plt.plot(self.asset_memory,'r')
#             plt.plot(account_growth)
#             plt.savefig('/home/ecsys/Documents/test.png'.format(iteration))
#             plt.close()
#             print("total_reward:{}".format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))- 10000 ))
#             return self.state, self.reward, self.terminal,{}

#         else:
#             # print(np.array(self.state[1:29]))


#             begin_total_asset = self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))
#             # print("begin_total_asset:{}".format(begin_total_asset))
#             argsort_actions = np.argsort(actions)
#             sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
#             buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

#             for index in sell_index:
#                 # print('take sell action'.format(actions[index]))
#                 self._sell_stock(index, actions[index])

#             for index in buy_index:
#                 # print('take buy action: {}'.format(actions[index]))
#                 self._buy_stock(index, actions[index])

#             self.day += 1
#             self.data = test_daily_data[self.day]
#             # self.money = self.state[0]
            


#             self.state =  [self.state[0]] + self.data.adjcp.values.tolist() + list(self.state[29:])
#             end_total_asset = self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))
#             # print("end_total_asset:{}".format(end_total_asset))
            
#             self.reward = end_total_asset - begin_total_asset            
#             # print("step_reward:{}".format(self.reward))

#             self.asset_memory.append(end_total_asset)


#         return self.state, self.reward, self.terminal, {}

#     def reset(self):
#         self.asset_memory = [10000]
#         self.day = 0
#         self.data = test_daily_data[self.day]
#         self.state = [10000] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        
#         # iteration += 1 
#         return self.state
    
#     def render(self, mode='human'):
#         return self.state

#     def _seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]




import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt

class StockTestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start_date='2018-04-02', end_date='2019-03-28', 
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

    def get_benchmark(self):
        benchmark = [self.start_balance]
        balance = self.start_balance

        date_pointer = self.get_index_from_date(self.start_date)
        cur_date = self.get_date_from_index(date_pointer[0])

        while cur_date < self.end_date:
            returns = []
            for idx, ticker in enumerate(self.tickers):
                ticker_row_idx = date_pointer[idx]
                today_price = self.stock_data[ticker].iloc[ticker_row_idx]['open']
                nextday_price = self.stock_data[ticker].iloc[ticker_row_idx-1]['open']
                returns.append( (nextday_price - today_price) / today_price )

            mean_return = np.mean(returns)
            benchmark.append(balance + balance * mean_return)
            balance = benchmark[-1]

            date_pointer = [date - 1 for date in date_pointer]
            cur_date = self.get_date_from_index(date_pointer[0])

        return benchmark

    def step(self, actions):

        # done if passed end_date
        date = self.get_date_from_index(self.date_pointer[0])
        if date >= self.end_date:
            self.terminal = True

        if self.terminal:
            benchmark = self.get_benchmark()
            plt.plot(self.asset_list,'r')
            plt.plot(benchmark)
            plt.savefig('/home/ecsys/Documents/test{}.png'.format(self.iteration))
            plt.close()
            print("done.", self.get_date_from_index(self.date_pointer[0]))
            return self.state, self.reward, self.terminal, {}

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