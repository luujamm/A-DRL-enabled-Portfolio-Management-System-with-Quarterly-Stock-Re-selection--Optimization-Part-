import numpy as np
import pandas as pd

from environment.portfolio_new import PortfolioEnv
from utils.define_args import define_args
from utils.data import *
from utils.yahoodownloader import get_data
from utils.draw import draw_test_figs, show_test_results, show_val_results
from utils.evaluation import evaluation_metrics
from utils.recorder import Recorder
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier


DATA = 'data/cb5_2_0410'
TARGET_NUM = 20

def mv_weights(year, Q):
    path = './' + DATA + '/' + 'train' + '/' + str(year) + 'Q' + str(Q) + '.csv'
    df = pd.read_csv(path)
    df = df[['date', 'close', 'tic']]
    df = df.pivot(index='date', columns='tic', values='close')
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    return weights


def test(args, year, Q, test_start_date):
    target_stocks = get_targets(year=year, Q=Q, num=TARGET_NUM)
    weights = mv_weights(year, Q)
    weights = list(weights.values())
    weights.insert(0, 0)

    test_history, test_dating = get_data(target_stocks, year, Q, 'test')
    tu_his, _ = get_data(target_stocks, year, Q, 'tu')
    test_data = test_history.copy()
    action_dim = TARGET_NUM + 1
    env = PortfolioEnv(args, test_history, test_data, action_dim, 
                           test_dating, tu_his, steps=50,
                           sample_start_date=test_start_date)
    env.reset()
    for t in itertools.count(start=1):
        weights, _, _, _, _, mv_trade_info, _ = env.step(weights, weights)
        


def main():
    years, quaters = get_years_and_quarters()
    quarter_dates = get_quarter_dates()
    args = define_args()
    for year in years:
        for Q in quaters:    
            key = str(year) + '_Q' + str(Q) 
            test_start_date = quarter_dates[key][2]
            test(args, year, Q, test_start_date)

            
            

            

if __name__ == '__main__':
    main()
