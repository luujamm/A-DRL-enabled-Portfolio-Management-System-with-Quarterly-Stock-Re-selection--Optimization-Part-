import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from path import test_path
from .data import get_years_and_quarters
from .evaluation import evaluation_metrics
from .yahoodownloader import get_data_repo


def load_value(dates, values, file):
    value = np.array(pickle.load(file))
    
    if len(dates) > 1:
        value *= values[-1]
    return np.concatenate((values, value))


def load_returns(dates, returns, file):
    return_ = np.array(pickle.load(file))
    return np.concatenate((returns, return_))


def render():
    test_dir = test_path()
    data_repo = get_data_repo()

    years, quarters = get_years_and_quarters()

    dates, ptfls, ews, sp500s, sp100s, ptfl_returns, ew_returns = [], [], [], [], [], [], []

    for year in years:
        for Q in quarters:
            with open(test_dir + str(year) + 'Q' + str(Q) + '/result.pickle', 'rb') as f:
                date = pickle.load(f)
                ptfls = load_value(dates, ptfls, f)
                ews = load_value(dates, ews, f)
                sp500s = load_value(dates, sp500s, f)
                sp100s = load_value(dates, sp100s, f)
                ptfl_returns = load_returns(dates, ptfl_returns, f)
                ew_returns = load_returns(dates, ew_returns, f)
                dates+=date
    
    with open('./' + data_repo + '/mv.pickle', 'rb') as f:
        mvs = pickle.load(f)

    with open(test_dir + 'result.pickle', 'wb') as f:
        pickle.dump(ptfls, f)
        pickle.dump(ews, f)
        pickle.dump(sp500s, f)
        pickle.dump(sp100s, f)
        pickle.dump(dates, f)


    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, ptfls, dates, ews, dates, mvs, dates, sp500s, dates, sp100s)
    fmt_year = mdates.AutoDateLocator()
    fmt = mdates.ConciseDateFormatter(fmt_year)
    fmt_month = mdates.DayLocator(interval=21)
    ax.xaxis.set_major_locator(fmt_year)
    ax.legend(['RL', 'EW', 'MV', 'S&P 500', 'S&P 100'])
    ax.set_ylabel('Cumulative Return')
    ax.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(test_dir + 'result.png')
    sharpe, sortino, mdd = evaluation_metrics(ptfl_returns, ptfls)
    print(ew_returns.shape)
    ew_sharpe, ew_sortino, ew_mdd = evaluation_metrics(ew_returns, ews)
    print('======\nOverall Evaluations:')
    print('RL: Sharpe = {:.3f}, Sortino = {:.3f}, MDD = {:.3f}'.format(sharpe, sortino, mdd))
    print('EW: Sharpe = {:.3f}, Sortino = {:.3f}, MDD = {:.3f}'.format(ew_sharpe, ew_sortino, ew_mdd))


