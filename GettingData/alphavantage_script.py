# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:41:25 2021

@author: rodri
"""

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time

key_path = "xxxx"

# extracting data for a single ticker
ts = TimeSeries(key=key_path, output_format='pandas')
data = ts.get_daily(symbol='EURUSD', outputsize='full')[0]
data.columns = ["open","high","low","close","volume"]
data = data.iloc[::-1] #order por data mais recente


# extracting stock data (historical close price) for multiple stocks
all_tickers = ["AAPL","MSFT","CSCO","AMZN","GOOG",
               "FB","BA","MMM","XOM","NKE","INTC"]
close_prices = pd.DataFrame()
api_call_count = 1
ts = TimeSeries(key=key_path, output_format='pandas')
start_time = time.time()
for ticker in all_tickers:
    data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='compact')[0]
    api_call_count+=1
    data.columns = ["open","high","low","close","volume"]
    data = data.iloc[::-1] #order por data mais recente
    close_prices[ticker] = data["close"]
    if api_call_count==5:
        api_call_count = 1
        time.sleep(60 - ((time.time() - start_time) % 60.0))