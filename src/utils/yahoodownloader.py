"""Contains methods and classes to collect data from
Yahoo Finance API
"""
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import csv


class YahooDownloader:
    def __init__(self, state_day_length, start_date: str, end_date: str, ticker_list: list):
        
        start_date = pd.to_datetime(start_date)
        start_date_shift = start_date + datetime.timedelta(days=-3*(state_day_length+20))
        shift_start_date = str(start_date_shift.date())
        self.shift_start_date = shift_start_date
        
        end_date = pd.to_datetime(end_date)
        end_date_shift = end_date + datetime.timedelta(days=2)
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
            temp_df["tic"] = tic
            data_df = pd.concat([data_df, temp_df], join='outer')
        
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