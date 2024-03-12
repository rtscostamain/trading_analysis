# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:16:45 2021

@author: rodri
"""

import datetime as dt
import yfinance as yf
import pandas as pd

stocks = ["AMZN", "MFST", "INTC", "GOOG", "INFY.NS"]
start = dt.datetime.today() - dt.timedelta(30)
end = dt.datetime.today()
cl_price = pd.DataFrame()
ohlvc_data = {}

for ticker in stocks:
    cl_price[ticker] = yf.download(ticker, start, end)["Adj Close"]
    
for ticker in stocks:
    ohlvc_data[ticker] = yf.download(ticker, start, end)
    
#ohlvc_data["AMZN"]["Open"]