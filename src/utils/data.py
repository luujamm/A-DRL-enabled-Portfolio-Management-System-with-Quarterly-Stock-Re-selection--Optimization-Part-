import numpy as np
import pandas as pd
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
    '2021_Q1': ['2017-10-02', '2020-10-01', '2021-01-04', '2021-04-01'],
    '2021_Q2': ['2018-01-02', '2021-01-04', '2021-04-01', '2021-07-01'],
    '2021_Q3': ['2018-04-02', '2021-04-01', '2021-07-01', '2021-10-01'],
    '2021_Q4': ['2018-07-02', '2021-07-01', '2021-10-01', '2022-01-01']
}


def get_targets(year, Q, num=None):
    start_year = 2018

    with open('./data/company_buy_5_2.pickle', 'rb') as f:
        all_target = pickle.load(f)

        for t in all_target:

            if 'PYPL' in t:
                t.remove('PYPL')

            if 'KHC' in t:
                t.remove('KHC')

    if num == None:
        return all_target[year - start_year + Q - 1]

    else:
        return all_target[year - start_year + Q - 1][:num]


def define_dates(args, year=None, Q=None):
    pretrain_start = '2002-01-01'
    turbulance_start = '2014-01-05'

    if args.case == 3:
        key = str(year) + '_Q' + str(Q) 
        quarter_dates = get_quarter_dates()
        train_start, train_end, val_end, test_end = (date for date in quarter_dates[key])

    else:
        raise ValueError('Case not defined')
        
    if args.test:
        return val_end, test_end

    elif args.backtest:
        return train_start, train_end

    else:
        return pretrain_start, turbulance_start, train_start, train_end, val_end

    
def normalize(x):
    return (x - 1) * 100


def state_normalizer(state):
    state = state / state[:, -1:, -1:] # normalize to last close
    state = normalize(state) 
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


def get_data_repo():
    repo = 'data/cb5_2_0410'
    return repo


def get_data(target_stocks, year, Q, status, bench=False):
    data_repo = get_data_repo()
    path = './' + data_repo + '/' + status + '/' + str(year) + 'Q' + str(Q)

    if bench:
        path += '_bench.csv'

    else:
        path += '.csv'

    df = pd.read_csv(path)
    df_hist = df[['date','open','high','low','close','volume','tic']]
    hist =[]
    
    for ticker in target_stocks:
        hist.append(np.expand_dims(df_hist[df_hist['tic']==ticker], axis=0))

    hist_np = np.concatenate(hist, axis=0)
    dating = hist_np[0][:,0]
    history = hist_np[:,:,1:5]
    num_training_time = len(dating)
    
    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[target_stocks.index(stock), :num_training_time, :]
    
    return target_history, dating