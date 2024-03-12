# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:37:52 2021

@author: rodri
"""

import pandas as pd
import numpy as np
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
import mplfinance as mpf

plt.rcParams['figure.figsize'] = [12, 7]

plt.rc('font', size=14)

name = 'SPY'
ticker = yfinance.Ticker(name)
df = ticker.history(interval="1d",start="2020-03-15",end="2020-07-15")
#df = ticker.history(interval="1b",period="1d")

df['Date'] = pd.to_datetime(df.index)
df['Date'] = df['Date'].apply(mpl_dates.date2num)

df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] \
    and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support
    
def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] \
    and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2] 
    return resistance

levels = []
for i in range(2,df.shape[0]-2):
  if isSupport(df,i):
    levels.append((i,df['Low'][i]))
  elif isResistance(df,i):
    levels.append((i,df['High'][i]))
    
def plot_all():
    fig, ax = plt.subplots()
    
    candlestick_ohlc(ax,df.values,width=0.6, colorup='green', colordown='red', alpha=0.8)
    
    date_format = mpl_dates.DateFormatter('%d-%m-%Y %H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    ax.set_ylabel('Price') 
    ax.grid(True)
    
    fig.tight_layout()
    
    for level in levels:
      plt.hlines(level[1],xmin=df['Date'][level[0]],\
                 xmax=max(df['Date']),colors='blue')
    fig.show()
    
plot_all()

s =  np.mean(df['High'] - df['Low'])

def isFarFromLevel(l):
    return np.sum([abs(l-x) < s  for x in levels]) == 0

levels = []
for i in range(2,df.shape[0]-2):
  if isSupport(df,i):
    l = df['Low'][i]

    if isFarFromLevel(l):
      levels.append((i,l))

  elif isResistance(df,i):
    l = df['High'][i]

    if isFarFromLevel(l):
      levels.append((i,l))

plot_all()


def plot_all_new(DF):
    fig, ax = plt.subplots()
    data = DF
    
    date_format = mpl_dates.DateFormatter('%d %b %Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    fig.tight_layout()
    
    for level in levels:
      plt.hlines(level[1],xmin=df['Date'][level[0]],\
                 xmax=max(df['Date']),colors='blue')
    fig.show()

    mpf.plot(data, type='candle', style='charles', title=name, ylabel='Price ($)', ylabel_lower='Shares \nTraded', mav=(3,6,9))    

plot_all_new(df)
