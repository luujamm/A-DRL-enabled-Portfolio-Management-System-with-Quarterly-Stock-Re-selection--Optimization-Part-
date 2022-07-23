"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""
import copy
import numpy as np

from src.utils.data import date_to_index, index_to_date
from src.utils.turbulence import tu_index


EPS = 1e-8


class DataGenerator(object):
    def __init__(self, args, history, state_data, action_dim, dating, steps=500, start_idx=0, start_date=None):
        assert history.shape[0] == action_dim - 1, 'Number of stock is not consistent'

        np.random.seed(args.seed)

        self.args = args
        self.steps = steps
        self.state_length = args.state_length
        self.start_idx = start_idx
        self.start_date = start_date
        self.dating = dating
        self.bias = 50 

        # make immutable class
        self._history_data = history.copy()
        
    def reset(self, epi_end_idx):
        self.step = 1
        
        # get data for this episode, each episode has different start_date.
        if self.start_date is None:
            self.idx = epi_end_idx-self.steps + self.bias
            
            print('Start date: {}'.format(index_to_date(self.idx, self.dating)))
            print('Episode_End_date: {}'.format(index_to_date(epi_end_idx, self.dating)))
            
            history_data = self._history_data[:, (self.idx - self.state_length):(self.idx + self.steps + 1), :]
            
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date, self.dating) - self.start_idx
            self.steps = self._history_data.shape[1] - self.idx - 1

            assert self.idx >= self.state_length and self.idx <= self._history_data.shape[1] - self.steps, \
                'Invalid start date, must be state_length day after start date and simulation steps day before end date'
            
            history_data = self._history_data[:, (self.idx - self.state_length):, :]
        
        self.history_data = history_data
        init_obs = self.history_data[:, self.step:(self.step + self.state_length), :].copy()
        init_ground_truth_obs = self.history_data[:, self.step + self.state_length:self.step + self.state_length + 1, :].copy()
        return init_obs, init_ground_truth_obs
               
    def _step(self):
        self.step += 1  
        obs = self.history_data[:, self.step:self.step + self.state_length, :].copy()
        # used for compute optimal action and sanity check
        ground_truth_obs = self.history_data[:, (self.step + self.state_length):(self.step + self.state_length + 1), :].copy()
        done = self.step >= self.steps
        return obs, done, ground_truth_obs


class PortfolioSim(object):
    def __init__(self, args, time_cost, steps):
        self.cost = args.trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.lam = args.lam
    
    def reset(self):
        self.infos = []
        self.p0 = 1.0

    def _step(self, w0, y0, w1, y1):      
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        p0 = self.p0 * np.dot(y0, w0)                               # ptfl value when open
        dw0 = (y0 * w0) / np.dot(y0, w0)                            # weight of open
        mu1 = self.cost * (np.abs(w1 - dw0)[1:]).sum()              # cost to change portfolio
        dw1 = (y1 * w1) / np.dot(y1, w1)                            # weight of close

        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)                        # final portfolio value
        p1 = p1 * (1 - self.time_cost)                              # we can add a cost to holding

        rho1 = p1 / self.p0 - 1                                     # rate of returns
        r1 = np.log((p1 + EPS) / (self.p0 + EPS))                   # log rate of return
        
        excess_ew_return = rho1 + 1 - np.mean((y0 * y1)[1:])
        reward = r1 + self.lam * excess_ew_return                   
        
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


class PortfolioEnv():
    def __init__(self, args, history, state_data, action_dim,
                 dating, tu_his, steps, time_cost=0.00, start_idx=0,
                 sample_start_date=None, epi_end_idx=None):
        self.state_length = args.state_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx
        self.dating = dating
        self.epi_end_idx = epi_end_idx
        self.tu_his = tu_his
        self.src = DataGenerator(args, history, state_data, action_dim, dating, steps=steps, start_idx=start_idx,
                                 start_date=sample_start_date)
        self.sim = PortfolioSim(args, time_cost=time_cost, steps=steps)
    
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
        np.testing.assert_almost_equal(action.squeeze().shape, (self.num_stocks + 1,) )
        
        # Check whether all action values are between 0 and 1 
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3, 
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
        y_close_to_open = open_price_vector / last_close_price_vector
        y_open_to_close = close_price_vector / open_price_vector
        
        new_weights, reward, excess_ew_return, info, done2 = self.sim._step(weights, y_close_to_open, action, y_open_to_close)
        tu = tu_index(observation, self.tu_his)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]

        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step, self.dating)
        
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs  
        self.infos.append(info)
        return new_weights, observation[1:, :, :], reward, excess_ew_return, done1 or done2, info, tu