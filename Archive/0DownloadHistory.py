import numpy as np
import pandas
import os
import warnings

#For Stock Data
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
from pprint import pprint

symbolList = ['CRWD', 'ADI', 'AMAT', 'ASML', 'AVGO', 'CREE', 'CY', 'ENTG', 'KLAC', 'LRCX',
                'MRVL', 'MXIM', 'MCHP', 'MKSI', 'MPWR', 'NVDA', 'NXPI', 'ON', 'QRVO', 'QCOM',
                'SLAB', 'SIMO', 'SWKS', 'TSM', 'TER', 'TXN', 'XLNX', 'GOOG', 'SBUX', 'SNE',
                'TSLA', 'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
                'DD', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 
                'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'V', 
                'WMT', 'MU', 'MRNA', 'NIO', 'GILD', 'AMD', 'AMZN', 'T', 'COST', 'NFLX',
                'PYPL']
symbL = ['MU']

def getData(stock):
    print(stock)
    #Outputting the Historical data into a .csv for later use
    ts = TimeSeries(key='MSR35IQGXL22FEMS', output_format='pandas')
    df_rev, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    df = df_rev.iloc[::-1]
    if os.path.exists('./Exports'):
        csv_name = ('Exports/' + stock + '_Export.csv')
    else:
        os.mkdir("Exports")
        csv_name = ('Exports/' + stock + '_Export.csv')
    df.to_csv(csv_name)
    time.sleep(12)

if __name__ == '__main__':
    for symb in symbL:
        getData(symb)