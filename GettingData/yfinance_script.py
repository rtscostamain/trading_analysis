# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:53:55 2021

@author: rodri
"""

import yfinance as yf

#data = yf.download("MSFT", period="6mo")
#data = yf.download("MSFT", start="2017-01-01", end="2021-01-01")
#data = yf.download("MSFT", start="2017-01-01", end="2021-01-01")
data = yf.download("MSFT", period="1mo", interval="5m")


msft = yf.Ticker("MSFT")

# get stock info
msft.info

# get historical market data
hist = msft.history(period="max")

# show actions (dividends, splits)
msft.actions

# show dividends
msft.dividends

# show splits
msft.splits

# show financials
msft.financials
msft.quarterly_financials

# show major holders
msft.major_holders

# show institutional holders
msft.institutional_holders

# show balance sheet
msft.balance_sheet
msft.quarterly_balance_sheet

# show cashflow
msft.cashflow
msft.quarterly_cashflow

# show earnings
msft.earnings
msft.quarterly_earnings

# show sustainability
msft.sustainability

# show analysts recommendations
msft.recommendations

# show next event (earnings, etc)
msft.calendar

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.isin

# show options expirations
msft.options

# get option chain for specific expiration
opt = msft.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts
