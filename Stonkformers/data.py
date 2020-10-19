import numpy as np
import pandas
import os
import warnings

#For Stock Data
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
from pprint import pprint

ts = TimeSeries(key='MSR35IQGXL22FEMS', output_format='pandas')

watchlist = ['TSLA', 'AAPL', 'AMZN', 'MSFT', 'GOOG', 'FB', 'AMD', 'NVDA']
symbL = ['TSLA']

def getData(stock):
    print(stock)
    # Outputting the Historical data into a .csv for later use
    # df_rev, meta_data = ts.get_daily_adjusted(symbol=stock, outputsize='full')
    df_rev, meta_data = ts.get_intraday(symbol=stock, outputsize='full')
    df = df_rev.iloc[::-1]
    if os.path.exists('./Data'):
        csv_name = ('Data/' + stock + '.csv')
    else:
        os.mkdir("Data")
        csv_name = ('Data/' + stock + '.csv')
    df.to_csv(csv_name)
    time.sleep(12)

if __name__ == '__main__':
    for symb in symbL:
        getData(symb)