import numpy as np


class Container(object):
    def __init__(self):
        self.daily_return = []
        self.values = []
        self.rewards = []
        self.weights = []
        self.date = []
        self.cost = 0
    
    def record_trades(self, action, trade_info):
        self.weights.append(action)
        self.values.append(trade_info['portfolio_value'])
        self.daily_return.append(trade_info['rate_of_return'])
        self.cost += trade_info['cost']
        
    def record_date(self, trade_info):
        self.date.append(trade_info['date'])

    def cal_benchmark_returns(self, values):
        b = np.array(self.values)
        benchmarks = np.array(self.values)[0, :, :values.shape[0], -1]
        benchmark_returns = benchmarks / benchmarks[:, :1]
        return benchmark_returns
        
    def cal_returns(self, test_num):
        values = np.array(self.values).reshape((test_num, -1))
        values = np.mean(values, axis=0)
        final_value = values[-1]    
        return final_value, values
    
    def cal_weights(self, action_dim, test_num):
        weights = np.mean(np.array(self.weights).T.reshape(action_dim, test_num, -1), 1)
        return weights
        
    def clear(self):
        del self.daily_return[:]
        del self.values[:]
        del self.rewards[:]
        del self.weights[:]
        del self.date[:]
        self.cost = 0


class Recorder(object):
    def __init__(self):
        self.train = Container()
        self.test = Container()
        self.ew = Container()
        self.benchmark = Container()
    
    def clear(self):
        self.train.clear()
        self.test.clear()
        self.ew.clear()
        

        
    
        
        