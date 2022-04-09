import numpy as np
import pickle

YEARS = [2018, 2019, 2020, 2021]
QUARTERS = [1, 2, 3, 4]
QUARTER_DATES = {
    '2018_Q1': ['2014-10-01', '2017-10-02', '2018-01-02', '2018-04-02'],
    '2018_Q2': ['2015-01-02', '2018-01-02', '2018-04-02', '2018-07-02'],
    '2018_Q3': ['2015-04-02', '2018-04-02', '2018-07-02', '2018-10-01'],
    '2018_Q4': ['2015-07-02', '2018-07-02', '2018-10-01', '2019-01-02'],
    '2019_Q1': ['2015-10-01', '2018-10-01', '2019-01-02', '2019-04-01'],
    '2019_Q2': ['2016-01-04', '2019-01-02', '2019-04-01', '2019-07-01'],
    '2019_Q3': ['2016-04-01', '2019-04-01', '2019-07-01', '2019-10-01'],
    '2019_Q4': ['2016-07-01', '2019-07-01', '2019-10-01', '2020-01-02'],
    '2020_Q1': ['2016-10-03', '2019-10-01', '2020-01-02', '2020-04-01'],
    '2020_Q2': ['2017-01-03', '2020-01-02', '2020-04-01', '2020-07-01'],
    '2020_Q3': ['2017-04-03', '2020-04-01', '2020-07-01', '2020-10-01'],
    '2020_Q4': ['2017-07-03', '2020-07-01', '2020-10-01', '2021-01-04'],
    '2021_Q1': ['2017-10-02', '2020-10-01', '2021-01-04', '2021-04-05'],
    '2021_Q2': ['2018-01-02', '2021-01-04', '2021-04-05', '2021-07-06'],
    '2021_Q3': ['2018-04-02', '2021-04-05', '2021-07-06', '2021-10-04'],
    '2021_Q4': ['2018-07-02', '2021-07-06', '2021-10-04', '2022-01-01']
}


def get_targets(year, Q, num):
    start_year = 2018
    with open('./utils/company_buy_0313.pickle', 'rb') as f:
        all_target = pickle.load(f)
        for t in all_target:
            if 'PYPL' in t:
                t.remove('PYPL')
            if 'KHC' in t:
                t.remove('KHC')
    return all_target[year - start_year + Q - 1][:num]


def define_dates(args, year=None, Q=None):
    pretrain_start = '2002-01-01'
    turbulance_start = '2014-01-05'
    if args.case == 3:
        key = str(year) + '_Q' + str(Q) 
        quarter_dates = get_quarter_dates()
        train_start, train_end, val_end, test_end = (date for date in quarter_dates[key])
    elif args.case == 4: #DP test4
        train_start = '2002-01-02'
        train_end, val_end, test_end = '2017-05-19', '2018-03-19', '2018-07-27'#'2021-01-12'
    elif args.case == 5: #DP test3
        train_start = '2010-01-04'
        train_end, val_end, test_end = '2015-04-06', '2016-02-01', '2016-06-09'#'2021-01-12'
    elif args.case == 6:
        train_start = '2006-05-06'
        train_end, val_end, test_end = '2011-12-06', '2012-10-03', '2013-02-14'#'2021-01-12'
    else:
        raise ValueError('Case not defined')
        
    if args.test:
        return val_end, test_end
    elif args.backtest:
        return train_start, train_end
    else:
        return pretrain_start, turbulance_start, train_start, train_end, val_end

    
def normalize(x):
    """ Create a universal normalization function across close/open ratio
    Args:
        x: input of any shape
    Returns: normalized data
    """
    return (x - 1) * 100


def state_normalizer(state):
    """ Preprocess observation obtained by environment
    Args:
        observation: (nb_classes, window_length, num_features) or with info
    Returns: normalized
    """
    state = state / state[:, -1:, -1:] # normalize to last close
    #state = state / state[:, :, -1:] # test 1
    #state = np.concatenate((state[:, :1, :], state), axis=1) # test 2
    #state = state[:, 1:, :] / state[:, :-1, -1:] # test 2
    state = normalize(state) # do "(x - 1)*100"
    return state


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
    data = history.copy()
        
    return history, dating, data


def generate_state(observation):
    cash = cash_state(observation)
    state = np.concatenate((cash, observation), 0)
    state = state_normalizer(state)
    return state


def cash_state(obs):
    cash = np.ones((1, obs.shape[1], obs.shape[2]))
    rfr = np.array([1.0])
    rate = np.exp(np.log(rfr) / 365)
    for i in range(cash.shape[1]-1):
        cash[:, i+1:, :] *= rate
    return cash


def get_quarter_dates():
    return QUARTER_DATES


def get_years_and_quarters():
    return YEARS, QUARTERS


def index_to_date(index, dating):
    return dating[index]    


def date_to_index(date_string, dating):
    date_idx = np.where(dating == date_string)
    return date_idx[0][0]  



#data = company_buy.pickle
'''old_dates = quater_dates = {
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
}'''