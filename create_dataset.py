import csv
import datetime
import numpy as np
import os
import pandas as pd
import time
import yfinance as yf

from src.utils.data import *
from src.utils.yahoodownloader import YahooDownloader


STATE_LENGTH = 100


def download_and_save(dir, year, Q, start, end, target_stocks, status):
    df = YahooDownloader(STATE_LENGTH,
                        start_date = start,
                        end_date = end,
                        ticker_list = target_stocks).fetch_data()

    df_hist = df[['date','open','high','low','close','volume','tic']]
    df_hist.to_csv(dir + '/' + status + '/' + str(year) + 'Q' + str(Q) + '.csv')

    if status == 'test' or 'val':
        # fetch benchmark data
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
            target_stocks = get_targets(year=year, Q=Q)
            
            download_and_save(dir, year, Q, tu_start, train_end, target_stocks, 'tu')
            download_and_save(dir, year, Q, train_start, train_end, target_stocks, 'train')
            download_and_save(dir, year, Q, train_end, val_end, target_stocks, 'val')
            download_and_save(dir, year, Q, val_end, test_end, target_stocks, 'test')
            print('Done')
    

if __name__ == '__main__':
    create_dataset()

            





