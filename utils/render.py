import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from path import test_path

def load_value(dates, values, file):
    value = np.array(pickle.load(file))
    
    if len(dates) > 1:
        value *= values[-1]
    return np.concatenate((values, value))

def render():
    test_dir = test_path()

    years = [2018, 2019, 2020, 2021]
    quarters = [1, 2, 3, 4]

    dates, ptfls, ews, sp500s, sp100s = [], [], [], [], []

    for year in years:
        for Q in quarters:
            with open(test_dir + str(year) + 'Q' + str(Q) + '/result.pickle', 'rb') as f:
                date = pickle.load(f)
                ptfls = load_value(dates, ptfls, f)
                ews = load_value(dates, ews, f)
                sp500s = load_value(dates, sp500s, f)
                sp100s = load_value(dates, sp100s, f)
                dates+=date

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, ptfls, dates, ews, dates, sp500s, dates, sp100s)
    fmt_year = mdates.AutoDateLocator()
    fmt = mdates.ConciseDateFormatter(fmt_year)
    fmt_month = mdates.DayLocator(interval=21)
    ax.xaxis.set_major_locator(fmt_year)
    ax.legend(['RL', 'EW', 'S&P 500', 'S&P 100'])
    ax.set_ylabel('Cumulative Return')
    ax.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(test_dir + 'result.png')
       