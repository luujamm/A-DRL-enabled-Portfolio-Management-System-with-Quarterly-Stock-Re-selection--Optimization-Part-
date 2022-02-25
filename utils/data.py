"""
Contains a set of utility function to process data
"""

import numpy as np
import pickle
from .yahoodownloader import get_history


def get_targets(year, Q, num):
    start_year = 2018
    with open('./utils/company_buy.pickle', 'rb') as f:
        all_target = pickle.load(f)
    
    return all_target[year - start_year + Q - 1][:num]


quater_dates = {
    '2018_Q1': ['2014-11-12', '2017-07-21', '2018-01-02', '2018-04-02'],
    '2018_Q2': ['2015-02-17', '2017-10-20', '2018-04-02', '2018-07-02'],
    '2018_Q3': ['2015-05-15', '2018-01-19', '2018-07-02', '2018-10-01'],
    '2018_Q4': ['2015-08-13', '2018-04-20', '2018-10-01', '2019-01-02'],
    '2019_Q1': ['2015-11-11', '2018-07-20', '2019-01-02', '2019-04-01'],
    '2019_Q2': ['2016-02-16', '2018-10-19', '2019-04-01', '2019-07-01'],
    '2019_Q3': ['2016-05-13', '2019-01-18', '2019-07-01', '2019-10-01'],
    '2019_Q4': ['2016-08-12', '2019-04-18', '2019-10-01', '2020-01-02'],
    '2020_Q1': ['2016-11-11', '2019-07-19', '2020-01-02', '2020-04-01'],
    '2020_Q2': ['2017-02-14', '2019-10-21', '2020-04-01', '2020-07-01'],
    '2020_Q3': ['2017-05-15', '2020-01-21', '2020-07-01', '2020-10-01'],
    '2020_Q4': ['2017-08-15', '2020-04-21', '2020-10-01', '2021-01-04'],
    '2021_Q1': ['2017-11-13', '2020-07-21', '2021-01-04', '2021-04-05'],
    '2021_Q2': ['2018-02-14', '2020-10-20', '2021-04-05', '2021-07-06'],
    '2021_Q3': ['2018-05-15', '2021-01-21', '2021-07-06', '2021-10-04'],
    '2021_Q4': ['2018-08-13', '2021-04-21', '2021-10-04', '2022-01-01']
}





def define_dates(args, year=None, Q=None):
    pretrain_start = '2002-01-01'
    turbulance_start = '2014-01-05'
    # 有往前多撈了2*state_day_length天的data
    if args.case == 3:
        key = str(year) + '_Q' + str(Q)
        #train_start = '2015-01-05'
        #train_end, val_end, test_end = '2017-07-21', '2018-01-04', '2018-06-09' 
        #train_start = '2015-04-05'
        #train_end, val_end, test_end = '2017-10-20', '2018-04-02', '2018-08-09' 
        train_start, train_end, val_end, test_end = (date for date in quater_dates[key])
    elif args.case == 4: #DP test4
        train_start = '2002-01-02'
        train_end, val_end, test_end = '2017-05-19', '2018-03-19', '2018-07-27'#'2021-01-12'
    elif args.case == 5: #DP test3
        train_start = '2002-01-02'
        train_end, val_end, test_end = '2015-04-06', '2016-02-01', '2016-06-09'#'2021-01-12'
    elif args.case == 6:
        train_start = '2009-05-06'
        train_end, val_end, test_end = '2011-12-06', '2012-10-03', '2013-02-14'#'2021-01-12'
    else:
        raise ValueError('Case not defined')
        
    if args.test:
        return val_end, test_end
    elif args.backtest:
        return train_start, train_end
    else:
        return pretrain_start, turbulance_start, train_start, train_end, val_end


'''def get_train_val_history(args, target_stocks, year=None, Q=None):
    pretrain_start, turbulance_start, train_start, train_end, val_end = define_dates(args, year, Q)
    history, dating = get_history(target_stocks, args.state_length, 
                                              turbulance_start, val_end) 
    
    shift = np.argwhere(dating==turbulance_start)[0][0]
    print(shift)
    train_start_idx = np.argwhere(dating==train_start)[0][0]
    train_end_idx = np.argwhere(dating==train_end)[0][0]
    turbulance_history = history[:, :train_end_idx+1, :]
    print(turbulance_history.shape)
    train_history, train_dating = history[:, train_start_idx-shift: train_end_idx+1, :], dating[train_start_idx-shift: train_end_idx+1]
    print(train_history.shape, train_dating[0], train_dating[-1])
    val_history, val_dating = history[:, train_end_idx-shift:, :], dating[train_end_idx-shift:]
    print(val_history.shape, val_dating[0], val_dating[-1])'''
    
    

    
def normalize(x):
    """ Create a universal normalization function across close/open ratio
    Args:
        x: input of any shape
    Returns: normalized data
    """

    return (x - 1) * 100


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment
    Args:
        observation: (nb_classes, window_length, num_features) or with info
    Returns: normalized
    """

    observation = observation[:, 1:, :] / observation[:, -1:, -1:] # normalize to last close
    #observation = observation[:, 1:, :4] / observation[:, :-1, -1:] # normalize to previous close
    
    observation = normalize(observation) # do "(x - 1)*100"

    return observation#/norm


def get_init_action(dim, random=False, ew=False):
    if random:
        init_action = np.random.rand(dim)
        init_action /= np.sum(init_action)
    elif ew:
        init_action = np.ones(dim)
        init_action[0] = 0
        init_action /= np.sum(init_action)
    else:
        init_action = np.zeros(dim)
        init_action[0] = 1
    return init_action


def transform_data(args, history, dating):
    data = None
    if args.closeae:    
        data = history.copy()
    else:
        data, dating = feature_convert(history, dating)
        history = history[:, -len(dating):, :]
        
    return history, dating, data


def transform_state(args, ae, state, observation):
    #cash = np.ones((1, state.shape[1], state.shape[2]))
    cash = cash_state(state)
    state = np.concatenate((cash, state), 0)
    
    
    if args.closeae:
        if observation.shape[1] == args.state_length:
            observation = np.concatenate((observation[:, 1:2, :], observation), axis=1)
        state = obs_normalizer(observation[:,:,:])  # remove cash element and normalize
                
    else:
        state = ae.extract(state)
    
    return state

def cash_state(state):
    cash = np.ones((1, state.shape[1], state.shape[2]))
    rfr = np.array([1.01])
    rate = np.exp(np.log(rfr) / 365)
    for i in range(cash.shape[1]-1):
        cash[:, i+1:, :] *= rate
    return cash
    