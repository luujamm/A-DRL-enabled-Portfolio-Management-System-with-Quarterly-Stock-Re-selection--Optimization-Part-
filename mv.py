import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

from src.environment.portfolio_env import PortfolioEnv
from src.utils.define_args import define_args
from src.utils.data import *
from src.utils.yahoodownloader import get_data, get_data_repo
from src.utils.draw import draw_test_figs, show_test_results, show_val_results
from src.utils.evaluation import evaluation_metrics
from src.utils.recorder import Container


data_repo = get_data_repo()
TARGET_NUM = 20


def mv_weights(year, Q):
    path = './' + data_repo + '/' + 'train' + '/' + str(year) + 'Q' + str(Q) + '.csv'
    df = pd.read_csv(path)
    df = df[['date', 'close', 'tic']]
    df = df.pivot(index='date', columns='tic', values='close')
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    return weights


def test(args, year, Q, test_start_date):
    target_stocks = get_targets(year=year, Q=Q, num=None)
    #weights = mv_weights(year, Q)
    #weights = np.array(list(weights.values()))
    target_stocks = target_stocks[:20]
    TARGET_NUM = len(target_stocks)
    
    # for equal weight
    weights = np.ones(TARGET_NUM) / TARGET_NUM
    # for equal weight

    weights = np.insert(weights, 0, 0)
    test_history, test_dating = get_data(target_stocks, year, Q, 'test')
    tu_his, _ = get_data(target_stocks, year, Q, 'tu')
    test_data = test_history.copy()
    action_dim = TARGET_NUM + 1
    
    env = PortfolioEnv(args, test_history, test_data, action_dim, 
                           test_dating, tu_his, steps=args.test_period_length,
                           sample_start_date=test_start_date)
    env.reset()

    mv_recorder = Container()

    for t in itertools.count(start=1):
        weights, _, _, _, done, mv_trade_info, _ = env.step(weights, weights)
        mv_recorder.record_trades(weights, mv_trade_info)
        mv_recorder.record_date(mv_trade_info)

        if done: 
            break
    
    mv_return = mv_recorder.cal_returns(1)[1]

    return mv_recorder.date, mv_return, mv_recorder.daily_return


def save(dates, returns):
    with open('./' + data_repo + '/ew_g1.pickle', 'wb') as f:
        pickle.dump(returns, f)
    

def main():
    years, quaters = get_years_and_quarters()
    quarter_dates = get_quarter_dates()
    args = define_args()
    dates, mv_returns, mv_daily_returns = [], np.array([]), []

    for year in years:
        for Q in quaters:    
            key = str(year) + '_Q' + str(Q) 
            test_start_date = quarter_dates[key][2]
            date, mv_return, mv_daily_return = test(args, year, Q, test_start_date)
            dates += date
            if len(mv_returns) > 0:
                mv_return *= mv_returns[-1]
            mv_returns = np.concatenate((mv_returns, mv_return))
            mv_daily_returns += mv_daily_return
            
    sharpe, sortino, mdd = evaluation_metrics(np.array(mv_daily_returns), mv_returns)
    print('MV Portfolio Value {:.5f}, SR = {:.3f}, StR = {:.3f}, MDD = {:.3f}'
          .format(mv_returns[-1], sharpe, sortino, mdd))
    save(dates, mv_returns)
            

if __name__ == '__main__':
    main()
