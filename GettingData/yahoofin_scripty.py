# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:45:49 2021

@author: rodri
"""

import pandas as pd
from yahoofinancials import YahooFinancials
import datetime as dt

all_tickers = ["AMZN", "MFST", "INTC", "GOOG", "INFY.NS"]

close_prices = pd.DataFrame()
end_date = (dt.date.today()).strftime('%Y-%m-%d')
beg_date = (dt.date.today()-dt.timedelta(1825)).strftime('%Y-%m-%d')
for ticker in all_tickers:
    yahoo_financials = YahooFinancials(ticker)
    json_obj = yahoo_financials.get_historical_price_data(beg_date, end_date, "daily")
    ohlv = json_obj[ticker]["prices"]
    temp = pd.DataFrame(ohlv)[["formatted_date", "adjclose"]]
    temp.set_index("formatted_date", inplace=True)
    temp.dropna(inplace=True)
    close_prices[ticker] = temp["adjclose"]