# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 23:13:29 2021

@author: rodri
"""

import requests

tickers = ["GME","AAPL","MSFT","CSCO","AMZN","GOOG",
               "FB","BA","MMM","XOM","NKE","INTC"] #list of tickers whose financial data needs to be extracted
financial_dir = {}

for ticker in tickers:
    url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'+ticker+'?formatted=true&crumb=8ldhetOu7RJ&lang=en-US&region=US&modules=defaultKeyStatistics%2CfinancialData%2CcalendarEvents&corsDomain=finance.yahoo.com'
    r = requests.get(url)
    data = r.json()
    
    financial_data = data['quoteSummary']['result'][0]
    
    financial_dir[ticker] = financial_data
    
    
