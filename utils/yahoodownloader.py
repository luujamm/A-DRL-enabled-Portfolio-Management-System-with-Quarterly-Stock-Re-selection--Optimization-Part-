"""Contains methods and classes to collect data from
Yahoo Finance API
"""
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import csv
# from utils.data_new import date_to_index, index_to_date


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, state_day_length, start_date: str, end_date: str, ticker_list: list):
        
        start_date = pd.to_datetime(start_date)
        start_date_shift = start_date + datetime.timedelta(days=-3*(state_day_length+20))
        shift_start_date = str(start_date_shift.date())
        self.shift_start_date = shift_start_date
        
        end_date = pd.to_datetime(end_date)
        end_date_shift = end_date + datetime.timedelta(days=2) #shift2天是因為最後值不被included，以及obs_ground_truth
        shift_end_date = str(end_date_shift.date())
        self.shift_end_date = shift_end_date
        
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.shift_start_date, end=self.shift_end_date, progress=False)
            # temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            temp_df["tic"] = tic
            #data_df = data_df.append(temp_df)

            data_df = pd.concat([data_df, temp_df], join='outer')
        # reset the index, we want to use numbers as index instead of dates
        
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            # data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
         # create day of the week column (monday = 0)
        data_df['day'] = data_df['date'].dt.dayofweek       
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        #print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())
        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df


'''def get_history(abbreviation, state_day_length, START_DATE, END_DATE):
    target_stocks = abbreviation
    df = YahooDownloader(state_day_length,
                         start_date = START_DATE,
                         end_date = END_DATE,
                         ticker_list = target_stocks).fetch_data()
    
    df_hist = df[['date','open','high','low','close','volume','tic']]
    
    #df_hist[['open','high','low','close']] = np.round(df_hist[['open','high','low','close']],2)
    
    hist =[]
    
    for ticker in target_stocks:
        hist.append(np.expand_dims(df_hist[df_hist['tic']==ticker], axis=0))
    #try:
    hist_np = np.concatenate(hist, axis=0)
    #except: pass
    dating = hist_np[0][:,0]
    #for i in range(hist_np.shape[1]-250):
    #    print(hist_np[0][i])
    #exit()
    
    history = hist_np[:,:,1:5]
    num_training_time = len(dating)
    
    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[target_stocks.index(stock), :num_training_time, :]
    
    return target_history, dating'''

def get_data_repo():
    #repo = 'data/cb5_2_0410'
    repo = 'data/ew'
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

    


