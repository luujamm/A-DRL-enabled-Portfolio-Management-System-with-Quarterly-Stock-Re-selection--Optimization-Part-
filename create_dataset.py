import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import csv
import os
import time
from utils.data import *
from utils.yahoodownloader import YahooDownloader

STATE_LENGTH = 40

'''quater_dates = {
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
}'''


def download_and_save(dir, year, Q, start, end, target_stocks, status):
    df = YahooDownloader(STATE_LENGTH,
                        start_date = start,
                        end_date = end,
                        ticker_list = target_stocks).fetch_data()

    df_hist = df[['date','open','high','low','close','volume','tic']]
    df_hist.to_csv(dir + '/' + status + '/' + str(year) + 'Q' + str(Q) + '.csv')
    if status == 'test' or 'val':
        df = YahooDownloader(STATE_LENGTH,
                        start_date = start,
                        end_date = end,
                        ticker_list = ['^GSPC', '^OEX']).fetch_data()
        df_hist = df[['date','open','high','low','close','volume','tic']]
        df_hist.to_csv(dir + '/' + status + '/' + str(year) + 'Q' + str(Q) + '_bench.csv')


def create_dataset():
    date = time.strftime('%Y-%m-%d', time.localtime())
    dir = './data/' + date
    if not os.path.exists(dir):
        os.mkdir(dir)
        os.mkdir(dir + '/tu')
        os.mkdir(dir + '/train')
        os.mkdir(dir + '/val')
        os.mkdir(dir + '/test')
    years, quarters = get_years_and_quarters()
    tu_start = '2014-01-05'
    
    for year in years:
        for Q in quarters:      
            print('Creating {}Q{}'.format(year, Q))      
            key = str(year) + '_Q' + str(Q)
            quarter_dates = get_quarter_dates()
            train_start, train_end, val_end, test_end = (date for date in quarter_dates[key])
            target_stocks = get_targets(year=year, Q=Q, num=20)

            download_and_save(dir, year, Q, tu_start, train_end, target_stocks, 'tu')
            download_and_save(dir, year, Q, train_start, train_end, target_stocks, 'train')
            download_and_save(dir, year, Q, train_end, val_end, target_stocks, 'val')
            download_and_save(dir, year, Q, val_end, test_end, target_stocks, 'test')
            print('Done')


if __name__ == '__main__':
    create_dataset()

            





