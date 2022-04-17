import numpy as np
import pandas as pd

from environment.portfolio_new import PortfolioEnv
from utils.data import *
from utils.yahoodownloader import get_data
from utils.draw import draw_test_figs, show_test_results, show_val_results
from utils.evaluation import evaluation_metrics
from utils.recorder import Recorder
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier




def main():
    data_repo = 'data/cb5_2_0410'
    years, quaters = get_years_and_quarters()
    target_num = 20
    for year in years:
        for Q in quaters:
            target_stocks = get_targets(year=year, Q=Q, num=target_num)
            train_history, train_dating = get_data(target_stocks, year, Q, 'train')
            path = './' + data_repo + '/' + 'train' + '/' + str(year) + 'Q' + str(Q) + '.csv'
            df = pd.read_csv(path)
            df = df[['date', 'close', 'tic']]
            df = df.pivot(index='date', columns='tic', values='close')
            mu = mean_historical_return(df)
            S = CovarianceShrinkage(df).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            print(weights)

            exit()

if __name__ == '__main__':
    main()
