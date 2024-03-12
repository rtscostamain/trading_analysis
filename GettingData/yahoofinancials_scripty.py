# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:40:27 2021

@author: rodri
"""

from yahoofinancials import YahooFinancials

ticker = 'AAPL'
yahoo_financials = YahooFinancials(ticker)
data = yahoo_financials.get_historical_price_data("2018-01-01", "2020-12-01", "daily")