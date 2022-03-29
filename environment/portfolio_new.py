"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""
from __future__ import print_function

import numpy as np
import copy

from utils.data import date_to_index, index_to_date

EPS = 1e-8


def random_shift(x, fraction):
    """ Apply a random shift to a pandas series. """
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def scale_to_start(x):
    """ Scale pandas series so that it starts at one. """
    x = (x + EPS) / (x[0] + EPS)
    return x


def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + EPS)) / np.std(returns - rfr + EPS)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + EPS)


def tu_index(observation, tu_his):
    returns = observation[1:,1:,3] / observation[1:,:-1,3] - 1
    y = returns[:, -1:]
    tu_returns = tu_his[:,1:,3] / tu_his[:,:-1,3] - 1
    u = np.mean(tu_returns, axis=1, keepdims=True)
    diff = y - u
    cov = np.cov(tu_returns)
    cov_ = np.linalg.pinv(cov)
    tu = np.matmul(diff.T, np.matmul(np.linalg.pinv(cov), diff))
    
    return tu


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, args, history, state_data, action_dim, dating, steps=500, window_length=50, start_idx=0, start_date=None):
        """
        Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        assert history.shape[0] == action_dim - 1, 'Number of stock is not consistent'
        
        # np.random.seed(args.seed)
        np.random.seed(args.seed)
        
        self.steps = steps
        self.window_length = args.state_length
        self.start_idx = start_idx
        self.start_date = start_date
        self.dating = dating
        self.bias = 50 if args.closeae else 20 #讓train和test的起始idx相同
        # make immutable class
        self._history_data = history.copy()  # axis:[assets,dates,ohlc]
        
    def reset(self, epi_end_idx):
        self.step = 1
        
        # get data for this episode, each episode has different start_date.
        if self.start_date is None:
            self.idx = epi_end_idx-self.steps + self.bias
            
            print('Start date: {}'.format(index_to_date(self.idx, self.dating)))
            print('Episode_End_date: {}'.format(index_to_date(epi_end_idx, self.dating)))
            
            history_data = self._history_data[:, (self.idx - self.window_length):(self.idx + self.steps + 1), :]
            
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date, self.dating) - self.start_idx
            self.steps = self._history_data.shape[1] - self.idx - 1

            assert self.idx >= self.window_length and self.idx <= self._history_data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
            
            history_data = self._history_data[:, (self.idx - self.window_length):, :]
        
        # apply augmentation?
        self.history_data = history_data
        
        init_obs = self.history_data[:, self.step:(self.step + self.window_length), :].copy()
        init_ground_truth_obs = self.history_data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        return init_obs, init_ground_truth_obs
               

    def _step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        self.step += 1  
        obs = self.history_data[:, self.step:self.step + self.window_length, :].copy()
        #print(self.step, self.step + self.window_length)
        # normalize obs with open price

        # used for compute optimal action and sanity check
        ground_truth_obs = self.history_data[:, (self.step + self.window_length):(self.step + self.window_length + 1), :].copy()

        done = self.step >= self.steps
        return obs, done, ground_truth_obs



class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, args, time_cost, steps):
        self.cost = args.trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.lam1 = args.lam1
        self.lam2 = args.lam2
        
    
    def reset(self):
        self.infos = []
        
        self.p0 = 1.0
        

    def _step(self, w0, y0, w1, y1):
        """
        Step.
        
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'
        p0 = self.p0 * np.dot(y0, w0) # ptfl value when open 
        dw0 = (y0 * w0) / np.dot(y0, w0) # t-1 close to t open
        mu1 = self.cost * (np.abs(w1 - dw0)[1:]).sum() # cost to change portfolio
        dw1 = (y1 * w1) / np.dot(y1, w1)  # (eq7) weights evolve into

        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / self.p0 - 1  # rate of returns
        r1 = np.log((p1 + EPS) / (self.p0 + EPS))  # log rate of return
        excess_ew_return = rho1 + 1 - np.mean((y0 * y1)[1:])
        reward = r1 - self.lam1 * np.max(w1) + self.lam2 * excess_ew_return # penalty on centralized weight
        #print(r1, self.lam2 * excess_ew_return, reward)
        
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 <= 0.1
        
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y0 * y1,
            "rate_of_return": rho1,
            "cost": p0 * mu1 * np.dot(y1, w1) ,
        }
        self.infos.append(info)
        
        return dw1, reward, excess_ew_return, info, done


# class PortfolioEnv(gym.Env):
class PortfolioEnv():
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, args,
                 history,
                 state_data,
                 action_dim,
                 dating, 
                 tu_his,
                 steps,           
                 time_cost=0.00,
                 start_idx=0,
                 sample_start_date=None,
                 epi_end_idx=None
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """
        
        self.window_length = args.state_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx
        self.dating = dating
        self.epi_end_idx = epi_end_idx
        self.tu_his = tu_his
        
        self.src = DataGenerator(args, history, state_data, action_dim, dating, steps=steps, start_idx=start_idx,
                                 start_date=sample_start_date)

        self.sim = PortfolioSim(args,
                                time_cost=time_cost, steps=steps)
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset(self.epi_end_idx)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info
    
    def step(self, weights, action):
        return self._step(weights, action)

    def _step(self, weights, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        ##########''' Check and processing the previous action '''#########
        # Check whether the actual action space is the same as desired
        
        np.testing.assert_almost_equal(action.squeeze().shape, (self.num_stocks + 1,) )
        
        # Check whether all action values are between 0 and 1 
        # print(action)
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3,  #這個3是什麼？不懂
                                       err_msg='weights should sum to 1. action="%s"' % weights)
        
        
        ###############''' Run DataGenerator '''###############
        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, observation.shape[1], observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        

        # relative price vector of last observation day (close/open)
        last_close_price_vector = observation[:, -2, 3]
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]

        y0 = open_price_vector / last_close_price_vector
        y1 = close_price_vector / open_price_vector
        
        new_weights, reward, excess_ew_return, info, done2 = self.sim._step(weights, y0, action, y1)
        
        tu = tu_index(observation, self.tu_his)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step, self.dating)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs  #其實沒用到

        self.infos.append(info)

        return new_weights, observation[1:, :, :], reward, excess_ew_return, done1 or done2, info, tu
        # return pre_action, observation, reward, done1 or done2, info