import numpy as np


class Recorder(object):
    def __init__(self):
        self.ptfl_values = []
        self.eqwt_values = []
        self.benchmarks = []
        self.rewards = []
        self.daily_return = []
        self.ew_daily_return = []
        self.weights = []
        self.test_date = []
        self.cost = 0
    
    def test_record(self, action, trade_info, ew_trade_info):
        self.weights.append(action)
        self.ptfl_values.append(trade_info['portfolio_value'])
        self.daily_return.append(trade_info['rate_of_return'])
        self.ew_daily_return.append(ew_trade_info['rate_of_return'])
        self.cost += trade_info['cost']
        
    def test_record_once(self, trade_info):
        self.eqwt_values.append(trade_info['portfolio_value'])
        self.test_date.append(trade_info['date'])
        
    def cal_returns(self, test_num, action_dim):
        mean_final_ptfl_value, ptfl_value = self.mean_final_value(test_num)
        ptfl_return = np.mean(ptfl_value, axis=0)
        
        benchmarks = np.array(self.benchmarks)[0, :, :ptfl_return.shape[0], -1]
        benchmark_returns = benchmarks / benchmarks[:, :1]
        
        weights = np.mean(np.array(self.weights).T.reshape(action_dim, test_num, -1), 1)
        
        return mean_final_ptfl_value, ptfl_return, self.eqwt_values, benchmark_returns, weights, self.test_date
    
    def mean_final_value(self, test_num):
        ptfl_value = np.array(self.ptfl_values).reshape((test_num, -1))
        final_ptfl_value = ptfl_value[:, -1]
        return np.mean(final_ptfl_value), ptfl_value
        
    def clear(self):
        del self.ptfl_values[:]
        del self.eqwt_values[:]
        del self.rewards[:]
        del self.daily_return[:]
        del self.ew_daily_return[:]
        del self.weights[:]
        del self.test_date[:]
        self.cost = 0
        
    
        
        