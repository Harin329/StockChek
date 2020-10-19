from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import time
import os
from pprint import pprint

#ts = TimeSeries(key='ED9X78MN9AYNC7TW', output_format='pandas')
#data, meta_data = ts.get_intraday(symbol='TSLA',interval='5min', outputsize='full')
#pprint(data)
ti = TechIndicators(key='ED9X78MN9AYNC7TW', output_format='pandas')

symbolList = ['CRWD', 'ADI', 'AMAT', 'ASML', 'AVGO', 'CREE', 'CY', 'ENTG', 'KLAC', 'LRCX',
                'MRVL', 'MXIM', 'MCHP', 'MKSI', 'MPWR', 'NVDA', 'NXPI', 'ON', 'QRVO', 'QCOM',
                'SLAB', 'SIMO', 'SWKS', 'TSM', 'TER', 'TXN', 'XLNX', 'GOOG', 'SBUX', 'SNE',
                'TSLA', 'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS',
                'DD', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 
                'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'V', 
                'WMT', 'MU', 'MRNA', 'NIO', 'GILD', 'AMD', 'AMZN', 'T', 'COST', 'NFLX',
                'PYPL']
sybmL = ['AIZ']

for sym in sybmL:
    data5, meta_data5 = ti.get_ema(symbol=sym, interval='daily', time_period=5, series_type='close')
    time.sleep(12)
    pprint(data5.tail(1))
    data20, meta_data20 = ti.get_ema(symbol=sym, interval='daily', time_period=20, series_type='close')
    time.sleep(12)
    pprint(data20.tail(1))

    fiveTwenty = data5.tail(1).values[0] > data20.tail(1).values[0]

    print(sym + " " + str(fiveTwenty))




