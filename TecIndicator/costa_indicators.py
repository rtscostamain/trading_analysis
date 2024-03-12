# Import necesary libraries
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from finvizfinance.util import webScrap, numberCovert
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import math
from tabulate import tabulate
import requests

from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
import matplotlib.dates as mpl_dates
import pandas as pd

from discord_webhook import DiscordWebhook, DiscordEmbed
from finvizfinance.quote import finvizfinance

################################ Discord Integration ####################################
def NotifyOportunityStock(buy_info):
    webhook = DiscordWebhook(url='https://discord.com/api/webhooks/831195888138715246/KLOQFLmbU8FCuHcGfjRykjiizAoLzuju8RegDxF-pNRzYGoJs_XnmnFXfeUdnZmM086')
    
    title = buy_info["ticker"] + " at " + str(buy_info["boughtAt"]).split()[1]
    description = str(buy_info["boughtAt"]).split()[0]
    reason = buy_info["buyReason"]

    embed = DiscordEmbed(title=title, description=description, color=242424)
    #embed.set_author(name='Author Name', url='https://github.com/lovvskillz', icon_url='https://avatars0.githubusercontent.com/u/14542790')
    embed.set_footer(text=reason)
    
    study = str(buy_info["study"])
    boughtPrice = str("{:.2f}".format(buy_info["boughtPrice"]))
    stopLoss = str("{:.2f}".format(buy_info["stopLoss"]))
    reward = str("{:.2f}".format(buy_info["reward"]))
    qty = str(buy_info["qty"])
    boughtTotal = str(round(buy_info["qty"] * buy_info["boughtPrice"], 2))
    
    embed.set_timestamp()
    embed.add_embed_field(name='Study', value=study)
    embed.add_embed_field(name='Stock price', value=boughtPrice)
    embed.add_embed_field(name='Stop loss', value=stopLoss)
    embed.add_embed_field(name='Goal', value=reward)
    
    webhook.add_embed(embed)
    response = webhook.execute()


def NotifyBuyStock(buy_info):
    webhook = DiscordWebhook(url='https://discord.com/api/webhooks/814330374658326538/O6xsz2BWTCDT9UQVJWYZaHAuTJOaX6-GDzV4rgHag5EOhtfdfykQN2jHK5I-bXQ7uIm')
    
    title = "Buy " + buy_info["ticker"]
    description = str(buy_info["boughtAt"])
    reason = buy_info["buyReason"]

    embed = DiscordEmbed(title=title, description=description, color=242424)
    #embed.set_author(name='Author Name', url='https://github.com/lovvskillz', icon_url='https://avatars0.githubusercontent.com/u/14542790')
    embed.set_footer(text=reason)
    
    study = str(buy_info["study"])
    boughtPrice = str("{:.2f}".format(buy_info["boughtPrice"]))
    stopLoss = str("{:.2f}".format(buy_info["stopLoss"]))
    reward = str("{:.2f}".format(buy_info["reward"]))
    qty = str(buy_info["qty"])
    boughtTotal = str(round(buy_info["qty"] * buy_info["boughtPrice"], 2))
    
    embed.set_timestamp()
    embed.add_embed_field(name='Study', value=study)
    embed.add_embed_field(name='Qty', value=qty)
    embed.add_embed_field(name='Stock price', value=boughtPrice)
    embed.add_embed_field(name='Stop loss', value=stopLoss)
    embed.add_embed_field(name='Goal', value=reward)
    embed.add_embed_field(name='Bought', value=boughtTotal)
    
    webhook.add_embed(embed)
    response = webhook.execute()


def NotifySellStock(sell_info):
    webhook = DiscordWebhook(url='https://discord.com/api/webhooks/814330374658326538/O6xsz2BWTCDOT9UQVWYZaHAuTJOaX6-GDzV4rgHag5EOhtfdfykQN2jHK5I-bXQ7uIm')
    
    title = "Sell " + sell_info["ticker"]
    description = "Bought: {0} | Sold: {1}".format(str(sell_info["boughtAt"]), str(sell_info["soldAt"]))


    color = 15730953 if sell_info["profit"] <= 0 else 830727
    gain_lost = 'Lost Total' if sell_info["profit"] <= 0 else 'Gain Total'

    embed = DiscordEmbed(title=title, description=description, color=color)
    #embed.set_author(name='Author Name', url='https://github.com/lovvskillz', icon_url='https://avatars0.githubusercontent.com/u/14542790')
    
    
    study = str(sell_info["study"])
    boughtPrice = str("{:.2f}".format(sell_info["boughtPrice"]))
    soldPrice = str("{:.2f}".format(sell_info["soldPrice"]))
    soldReason = sell_info["soldReason"]
    profit = str("{:.2f}".format(sell_info["profit"]))
    qty = str(sell_info["qty"])
    percent = str(round(((sell_info["boughtPrice"]*100)/sell_info["soldPrice"]-100)*-1, 2)) + "%"
    profit_total = str(round(sell_info["qty"] * sell_info["profit"], 2))
    
    embed.set_timestamp()
    embed.add_embed_field(name='Study', value=study)
    embed.add_embed_field(name='Qty', value=qty)
    embed.add_embed_field(name='Bought price', value=boughtPrice)
    embed.add_embed_field(name='Sold price', value=soldPrice)
    embed.add_embed_field(name='Profit', value=profit)
    embed.add_embed_field(name='Percent', value=percent)
    embed.add_embed_field(name=gain_lost, value=profit_total)
    
    embed.set_footer(text=soldReason)
    
    webhook.add_embed(embed)
    response = webhook.execute()
    
def NotifyStartDayTrade(tickers_info):
    webhook = DiscordWebhook(url='https://discord.com/api/webhooks/814330374658326538/O6xsz2BWTCDOT9QVJWYZaHAuTJOaX6-GDzV4rgHag5EOhtfdfykQN2jHK5I-bXQ7uIm')
    
    now = dt.datetime.now()
    title = "Day Trade scan start"
    description = str(now.strftime("%m/%d/%Y, %H:%M"))

    embed = DiscordEmbed(title=title, description=description, color=242424)
    #embed.set_author(name='Author Name', url='https://github.com/lovvskillz', icon_url='https://avatars0.githubusercontent.com/u/14542790')
    
    tickerList = ""
    sentimentalList = ""
    stopLossList = ""
    industryList = ""
    sectorList = ""
    sectorIndustryList = ""
    gapList = ""
    BR_RE = '\n'
    
    for ticker in tickers_info:
        tickerList += str(ticker["ticker"]) + BR_RE
        industryList += str(ticker["industry"]) + BR_RE
        sectorList += str(ticker["sector"]) + BR_RE
        sectorIndustryList += str(ticker["sector"]) + "|" + str(ticker["industry"]) + BR_RE
        gapList += str(ticker["gap"]) + BR_RE
        sentimentalList += "no{0}".format(BR_RE)
        stopLossList += str(ticker["stop_loss_percent"]) + "%" + BR_RE
        
        
    embed.set_timestamp()
    embed.add_embed_field(name='Ticker', value=tickerList)
    embed.add_embed_field(name='Gap', value=gapList)
    embed.add_embed_field(name='Sector', value=sectorIndustryList)
    #embed.add_embed_field(name='Stop Loss', value=stopLossList)
    #embed.add_embed_field(name='Sentimental', value=sentimentalList)
    
    webhook.add_embed(embed)
    response = webhook.execute()    

def OportunitySend(info, tran_date):

    now = dt.datetime.now()
    now_ninus = now + dt.timedelta(minutes = -62) #122 = 2 horas por conta do fuso e 1, 62 = 1 hora
    #print("now_ninus", now_ninus)
    #print("tran_date", tran_date)
    
    sendNotification = tran_date > now_ninus
    #sendNotification = False

    if sendNotification:
        NotifyOportunityStock(info)


def NotificationSend(info, action, tran_date):

    now = dt.datetime.now()
    now_ninus = now + dt.timedelta(minutes = -62) #122 = 2 horas por conta do fuso e 1, 62 = 1 hora
    #print("now_ninus", now_ninus)
    #print("tran_date", tran_date)
    
    sendNotification = tran_date > now_ninus
    sendNotification = False

    if sendNotification:
        if (action == "sell"):
            NotifySellStock(info)
        else:
            NotifyBuyStock(info)

def TransactionSend(info, tran_date):

    now = dt.datetime.now()
    now_ninus = now + dt.timedelta(minutes = -62) #122 = 2 horas por conta do fuso e 1, 62 = 1 hora
    #print("now_ninus", now_ninus)
    #print("tran_date", tran_date)
    
    sendTransaction = tran_date > now_ninus
    sendTransaction = False
    
    if sendTransaction:
        result, status = send_investor_transaction(info)


'''
buy_info = {
    "ticker": "STNE",
    "date": "2021-11-78 11:50",
    "study": "RSI",
    "boughtPrice": 20.55,
    "stopLoss": 20.00,
    "reward": 25.00,
    "qty": 1,
    "buyReason": "RSI > 30 AND Profit > 5.00"
    }
NotifyBuyStock(buy_info)


sell_info = {
    "ticker": "STNE",
    "date": "2021-11-78 11:50",
    "study": "RSI",
    "soldReason": "Profit $$$",
    "boughtPrice": 10.50,
    "soldPrice": 11.50,
    "profit": -1.00,
    "qty": 1
    }
NotificationSend(sell_info, "sell")

'''

################################ functions ####################################
def Stochastic(DF, n):
    "A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time"
    df = DF.copy()
    df['STO_K'] = ((df['Close'] - df['Low'].rolling(window=n, center=False).mean()) / (df['High'].rolling(window=n, center=False).max() - df['Low'].rolling(window=n, center=False).min())) * 100
    df['STO_D'] = df['STO_K'].rolling(window = 3, center=False).mean()
    df["STO_K"] = df["STO_K"].fillna(0)
    df["STO_D"] = df["STO_D"].fillna(0)
    return df

def StochasticGraph(df):
    ax = df[['STO_K', 'STO_D']].plot()
    df['Adj Close'].plot(ax=ax, secondary_y=True)
    ax.axhline(20, linestyle='--', color="r")
    ax.axhline(80, linestyle="--", color="r")
    plt.show()


def VWAP(DF, multiple):
    "function to Calculation of Volume Weighted Average Price"
    df = DF.copy()
    df['VWAP'] = (df.Volume * (df.High + df.Low) / 2).cumsum() / df.Volume.cumsum()
    df["VWAP"] = df["VWAP"].fillna(0)
        
    df['VWAP_MEAN_DIFF'] = ((df.High + df.Low) / 2) - df.VWAP
    df['SQ_DIFF'] = df.VWAP_MEAN_DIFF.apply(lambda x: math.pow(x, 2))
    df['SQ_DIFF_MEAN'] = df.SQ_DIFF.expanding().mean()
    df['STDEV_TT'] = df.SQ_DIFF_MEAN.apply(math.sqrt)
    
    df['VWAP_DEV_UP'] = df.VWAP + multiple * df['STDEV_TT']
    df['VWAP_DEV_DN'] = df.VWAP - multiple * df['STDEV_TT']
    
    df2 = df.drop(['VWAP_MEAN_DIFF','SQ_DIFF','SQ_DIFF_MEAN', 'STDEV_TT'],axis=1)
    
    return df2

def VWAPGraph(DF):
    addplot  = [
        mpf.make_addplot(DF['VWAP']),
        mpf.make_addplot(DF['VWAP_DEV_UP']),
        mpf.make_addplot(DF['VWAP_DEV_DN']),
    ]
    
    mpf.plot(DF, type='candle', addplot=addplot)

def ATR(DF, n):  
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def ADR(DF, n):
    "function to calculate Average Daily Range"
    df = DF.copy()
    df['ADR_Div']=df['High']/df['Low']
    df_adr = df.tail(n)
    adr = 0.0
    index = 0
    
    for index, row in df_adr.iterrows():
        adr = adr + row['ADR_Div']
    
    adr_percent = round(100*((adr)/n-1),2)
    
    return adr_percent

def KeltnerChannel(DF, n):  
    """Calculate Keltner Channel for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    df = DF.copy()
    KelChM = pd.Series(((df['High'] + df['Low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(n, min_periods=n).mean(),
                       name='KelChD_' + str(n))

    df['Kelch_Upper'] = KelChU
    df['Kelch_Middle'] = KelChM
    df['Kelch_Down'] = KelChD
    return df

def BollingerBand(DF):
    """function to calculate Bollinger band
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA20"] = df["Adj Close"].rolling(window=20).mean()
    df["20dSTD"] = df["Adj Close"].rolling(window=20).std()
    
    df["Upper"] = df['MA20'] + (df['20dSTD'] * 2)
    df["Lower"] = df['MA20'] - (df['20dSTD'] * 2)
    #df.dropna(inplace=True)
    return df

def BollingerBandGraph(DF, ticker):
    """Closing prices above the upper Bollinger band may indicate that currently the stock price is too high and price may decrease soon.
       Closing prices below the lower Bollinger band may be seen as a sign that prices are too low and they may be moving up soon."""
    DF[['Adj Close','MA20','Upper','Lower']].plot(figsize=(10,4))
    plt.grid(True)
    plt.title(ticker + ' Bollinger Bands')
    plt.axis('tight')
    plt.ylabel('Price')
    

def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df.dropna(inplace=True)
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["MA_Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    #df.dropna(inplace=True)
    return df

def MACDGraph(DF, ticker):
    """MACD Graph"""
    # Visualization - Using object orient approach
    # Get the figure and the axes
    df = DF.copy()
    df.dropna(inplace=True)
    fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=False, figsize=(10, 6), gridspec_kw = {'height_ratios':[2.5, 1]})
    df.iloc[-100:,4].plot(ax=ax0)
    ax0.set(ylabel='Adj Close')
    
    df.iloc[-100:,[-2,-1]].plot(ax=ax1)
    ax1.set(xlabel='Date', ylabel='MACD/Signal')
    
    # Title the figure
    fig.suptitle(ticker + 'Stock Price with MACD', fontsize=14, fontweight='bold')

# Calculating RSI without using loop
def RSI(DF, n):
    """function to calculate Relative Strength Index (RSI)
    The RSI will then be a value between 0 and 100. It is widely accepted that when the RSI is 30 or below, the stock is undervalued and when it is 70 or above, the stock is overvalued.
    """
    df = DF.copy()
    delta = df["Adj Close"].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n-1]] = np.mean( u[:n]) # first value is average of gains
    u = u.drop(u.index[:(n-1)])
    d[d.index[n-1]] = np.mean( d[:n]) # first value is average of losses
    d = d.drop(d.index[:(n-1)])
    rs = u.ewm(com=n,min_periods=n).mean()/d.ewm(com=n,min_periods=n).mean()
    df["RSI"] = 100 - 100 / (1+rs)
    df["RSI"] = df["RSI"].fillna(0)
    return df

def RSIGraph(DF, ticker):
    DF[['Adj Close','RSI']].plot(figsize=(10,4))
    plt.grid(True)
    plt.title(ticker + ' RSI')
    plt.axis('tight')

    
def MovingAverage(DF, moving_avg, short_window, long_window):
    '''
    The function takes the stock symbol, time-duration of analysis, 
    look-back periods and the moving-average type(SMA or EMA) as input 
    and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
    
    # short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20 
    # long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200    
    '''
    df = DF.copy()
    
    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  
    
    
    if moving_avg == 'SMA':
        # Create a short simple moving average column
        df[short_window_col] = df['Close'].rolling(window = short_window, min_periods = 1).mean()
    
        # Create a long simple moving average column
        df[long_window_col] = df['Close'].rolling(window = long_window, min_periods = 1).mean()
    
    elif moving_avg == 'EMA':
        # Create short exponential moving average column
        df[short_window_col] = df['Close'].ewm(span = short_window, adjust = False).mean()
    
        # Create a long exponential moving average column
        df[long_window_col] = df['Close'].ewm(span = long_window, adjust = False).mean()
        
        
    # then set Signal as 1 else 0.
    df['Avg Signal'] = 0.0  
    df['Avg Signal'] = np.where(df[short_window_col] > df[long_window_col], 1.0, 0.0) 

    # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    df['Avg Position'] = df['Avg Signal'].diff()
    
    return df
    
def MovingAverageGraph(DF, stock_symbol, moving_avg, short_window, long_window):
    
    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  
    
    # plot close price, short-term and long-term moving averages
    plt.figure(figsize = (20,10))
    plt.tick_params(axis = 'both', labelsize = 14)
    DF['Close'].plot(color = 'k', lw = 1, label = 'Close')  
    DF[short_window_col].plot(color = 'b', lw = 1, label = short_window_col)
    DF[long_window_col].plot(color = 'g', lw = 1, label = long_window_col) 
    
    # plot 'buy' signals
    plt.plot(DF[DF['Avg Position'] == 1].index, 
            DF[short_window_col][DF['Avg Position'] == 1], 
            '^', markersize = 15, color = 'g', alpha = 0.7, label = 'buy')
    
    # plot 'sell' signals
    plt.plot(DF[DF['Avg Position'] == -1].index, 
            DF[short_window_col][DF['Avg Position'] == -1], 
            'v', markersize = 15, color = 'r', alpha = 0.7, label = 'sell')
    plt.ylabel('Price in U$', fontsize = 16 )
    plt.xlabel('Date', fontsize = 16 )
    plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()
 
def MovingAveragePrint(DF):
    df_pos = DF[(DF['Avg Position'] == 1) | (DF['Avg Position'] == -1)]
    df_pos['Avg Position'] = df_pos['Avg Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    print(tabulate(df_pos, headers = 'keys', tablefmt = 'psql'))
    
# Calculating Support / Resistence
def isSupport(df,i):
    support = df['Low'][i] < df['Low'][i-1]  and df['Low'][i] < df['Low'][i+1] \
    and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support
    
def isResistance(df,i):
    resistance = df['High'][i] > df['High'][i-1]  and df['High'][i] > df['High'][i+1] \
    and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2] 
    return resistance

def isFarFromLevel(l, s, levels):
    return np.sum([abs(l-x) < s  for x in levels]) == 0

def supportResistence(DF):
    df = DF.copy()
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    
    #Levels
    levels = []
    s =  np.mean(df['High'] - df['Low'])
    
    for i in range(2,df.shape[0]-2):
      if isSupport(df,i):
        l = df['Low'][i]
    
        if isFarFromLevel(l,s,levels):
          levels.append((i,l))
    
      elif isResistance(df,i):
        l = df['High'][i]
    
        if isFarFromLevel(l,s,levels):
          levels.append((i,l))
    return levels      

def supportResistenceGraph(DF, levels):
    df = DF.copy()
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    
    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    
    #Graph
    fig, ax = plt.subplots()
    
    candlestick_ohlc(ax,df.values,width=0.6, colorup='green', colordown='red', alpha=0.8)
    
    date_format = mpl_dates.DateFormatter('%d-%m-%Y  %H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    ax.set_ylabel('Price') 
    ax.grid(True)
    
    fig.tight_layout()

    for level in levels:
      plt.hlines(level[1],xmin=df['Date'][level[0]],\
                 xmax=max(df['Date']),colors='blue')
    fig.show()
    
# Stop Loss
def investiment_stop_loss(stock_price, percent):
    amount = (stock_price * (percent/100))
    return stock_price - amount

# Reward
def investiment_reward(stock_price, percent):
    amount = (stock_price * (percent/100))
    return stock_price + amount


################################ BuySell ####################################

def register_oportunity(userId, bought_at, ticker, boughtPrice, stopLoss, reward, qty, study, buyReason):
    
    result = {
        "userId": userId,
        "boughtAt": str(bought_at.replace(tzinfo=None)),
        "ticker": ticker,
        "study": study,
        "boughtPrice": round(boughtPrice,2),
        "stopLoss": round(stopLoss,2),
        "reward": round(reward,2),
        "qty": qty,
        "buyReason": buyReason,
        "type": "bought"
        }
    #print(result)
    
    try:
        OportunitySend(result, bought_at.replace(tzinfo=None))
    except Exception as e:
        print("Error: {0}".format(e))
    
    return result


def register_purchase(userId, bought_at, ticker, boughtPrice, stopLoss, reward, qty, study, buyReason, sendTransaction):
    
    result = {
        "userId": userId,
        "boughtAt": str(bought_at.replace(tzinfo=None)),
        "ticker": ticker,
        "study": study,
        "boughtPrice": round(boughtPrice,2),
        "stopLoss": round(stopLoss,2),
        "reward": round(reward,2),
        "qty": qty,
        "buyReason": buyReason,
        "type": "bought"
        }
    #print(result)
    
    try:
        NotificationSend(result, "buy", bought_at.replace(tzinfo=None))
        
        if sendTransaction == True:
            TransactionSend(result, bought_at.replace(tzinfo=None))
    except Exception as e:
        print("Error: {0}".format(e))
    
    return result

def register_sale(userId, sold_at, bought_at, ticker, boughtPrice, soldPrice, profit, qty, study, soldReason, sendTransaction):
    
    
    boughtAt = str(bought_at.replace(tzinfo=None))
    soldAt = str(sold_at.replace(tzinfo=None))
    boughtPrice = round(boughtPrice,2)
    soldPrice = round(soldPrice,2)
    profit = round(profit,2)
    
    result = {
        "userId": userId,
        "boughtAt": boughtAt,
        "soldAt": soldAt,
        "ticker": ticker,
        "study": study,
        "boughtPrice": boughtPrice,
        "soldPrice": soldPrice,
        "profit": profit,
        "qty": qty,
        "soldReason": soldReason,
        "type": "sold"
        }

    alreadysold = ((df_tickerBought['userid'] == userId) & 
                (df_tickerBought['ticker'] == ticker) &
                (df_tickerBought['boughtat'] == boughtAt) &
                (df_tickerBought['bought'] == boughtPrice)).any()
    
    if not alreadysold: 
        df_tickerBought.loc[len(df_tickerBought.index)] = [userId, ticker, boughtAt, boughtPrice]
        #print(df_tickerBought)

        try:
            NotificationSend(result, "sell", sold_at.replace(tzinfo=None))
            if sendTransaction:
                TransactionSend(result, sold_at.replace(tzinfo=None))
        except Exception as e:
            print("Error: {0}".format(e))
    
    #print(result)
    return result

def make_investment_calc_macd(ohlcv, params):
    
    macd_param = params["macd"]
    userId = params["user_id"]
    tickerInfo = params["ticker"]
    
    volume_greater = tickerInfo["volume_greater"]
    safe_profit_reward_percent = tickerInfo["reward_profit_saved_percent"]
    
    df_BuySell = ohlcv.copy()
    df_BuySell = MACD(df_BuySell, macd_param["fast_length"], macd_param["slow_length"], macd_param["macd_length"])
    
    first_stopLoss = 0.00
    actual_stopLoss = 0.00
    actual_reward = 0.00
    actual_signal = ""
    high_profit = 0.00
    
    total_earned = 0.00
        
    result_signal = []
    result_bought = []
    result_sold = []
    result_close = []
    result_profit = []
    result_macd_stop_loss = []
    result_macd_reward = []
    
    result_stock_sold = []
    
    bought_at = ""
    bought_price = 0.00
    sold_price = 0.00
    profit = 0.00
        

    for index, row in df_BuySell.iterrows():
        #calculando o meu risco
        stockPrice = round(row["Close"], 3)
        stopLoss = row["Stop Loss"]
        reward = row["Reward"]
        

        #MACD Buy/Sell Signal ======= BEGIN
        macd_value = row["MACD"]
        macd_signal = row["MA_Signal"]
        #Sem compra
        if (actual_signal == ""):
     
             #Voltou acima de 30
             if (macd_signal > 0) and (macd_value > macd_signal) and (row["Volume"] > volume_greater):
                actual_signal = "Buy"
                bought_price = stockPrice
                bought_at = index
                sold_price = 0.00
                profit = 0.00
                actual_stopLoss = stopLoss
                first_stopLoss = stopLoss
                actual_reward = reward
                
                reason = "Signal > 0 AND MACD ({0}) > Signal ({1}) AND Volume ({2}) > ({3})".format(macd_value, macd_signal, row["Volume"], volume_greater)
                register_purchase(userId,
                                bought_at,
                                params["ticker"],
                                bought_price,
                                stopLoss,
                                reward,
                                1, 
                                "MACD",
                                reason,
                                False)
                
             
                 
        elif (actual_signal == "Buy"):
            
            profit = (stockPrice - bought_price)
            
            if high_profit == 0:
                high_profit = profit
            
            #Sinal de venda, pode cair
            if (macd_value < macd_signal):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index,
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 "MACD", 
                                 "Signal > MACD",
                                 False)
                   )
               
            #calculo o Stoploss
            if (stockPrice <= first_stopLoss):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 "MACD", 
                                 "Stop loss U$" + str(first_stopLoss),
                                 False)
                   )
               
               
            #calculo o novo reward
            if (stockPrice > actual_reward):
                actual_reward = reward
                actual_stopLoss = stopLoss
    
            #calculo o novo reward
            if (profit > high_profit):
                actual_reward = reward
                actual_stopLoss = stopLoss
                high_profit = profit
    
            #Garanto um lucro quando cair abaixo de 50% do que eu já ganhei
            if (profit > 0) and ((high_profit > 0) and (((profit*100) / high_profit) < safe_profit_reward_percent)) and (high_profit - profit > tickerInfo["reward_minimun_amount"]):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 "MACD", 
                                 "Stop loss - " + str(safe_profit_reward_percent) +"% Total earned"),
                                 False
                                 )
               
                    
                
        result_signal.append(actual_signal)
        result_bought.append(bought_price)
        result_sold.append(sold_price)
        result_profit.append(profit)
        result_macd_stop_loss.append(actual_stopLoss)
        result_close.append(stockPrice)
        result_macd_reward.append(actual_reward)
        
        if (actual_signal == "Sell"):
           actual_signal = ""
           bought_at = ""
           bought_price = 0.00
           sold_price = 0.00
           profit = 0.00
           actual_stopLoss = 0.00
           high_profit = 0.00
               
        #MACD Buy/Sell Signal ======= END


    df_BuySell["MACD Signal"] = result_signal
    df_BuySell["MACD Bought"] = result_bought
    df_BuySell["MACD Sold"] = result_sold
    df_BuySell["MACD Close"] = result_close
    df_BuySell["MACD Profit"] = result_profit
    df_BuySell["MACD Stop Loss"] = result_macd_stop_loss
    df_BuySell["MACD Reward"] = result_macd_reward

    return df_BuySell, total_earned, result_stock_sold


def make_investment_calc_rsi(ohlcv, params):
    
    rsi_param = params["rsi"]
    userId = params["user_id"]
    tickerInfo = params["ticker"]
    
    df_BuySell = ohlcv.copy()
    df_BuySell = RSI(df_BuySell, 14)
    
    first_stopLoss = 0.00
    actual_stopLoss = 0.00
    actual_reward = 0.00
    actual_signal = ""
    high_profit = 0.00
    
    recovery_above_30 = rsi_param["recovery_above_30"]
    
    volume_greater = tickerInfo["volume_greater"]
    safe_profit_reward_percent = tickerInfo["reward_profit_saved_percent"]
    
    total_earned = 0.00
    
    result_signal = []
    result_bought = []
    result_sold = []
    result_close = []
    result_profit = []
    result_rsi_stop_loss = []
    result_rsi_reward = []
    
    result_stock_sold = []
    
    bought_at = ""
    bought_price = 0.00
    sold_price = 0.00
    profit = 0.00
    below_value = False


    for index, row in df_BuySell.iterrows():
        stockPrice = round(row["Close"], 3)
        stopLoss = row["Stop Loss"]
        reward = row["Reward"]
        
        #RSI Buy/Sell Signal ======= BEGIN
        rsi_value = row["RSI"]
        #Sem compra
        if (actual_signal == ""):
             #Monitoro para saber se bateu abaixo de 30
             if (rsi_value < 30):
                 below_value = True
     
             #Voltou acima de 30
             if ((below_value == True) and (rsi_value > 30) and (rsi_value > recovery_above_30) and (row["Volume"] > volume_greater)):
                actual_signal = "Buy"
                bought_price = stockPrice
                bought_at = index
                sold_price = 0.00
                profit = 0.00
                actual_stopLoss = stopLoss
                first_stopLoss = stopLoss
                actual_reward = reward
                below_value = False
                
                reason = "RSI was bellow 30 AND RSI ({0}) > ({1}) AND Volume ({2}) > ({3})".format(rsi_value, recovery_above_30, row["Volume"], volume_greater)
                register_purchase(userId,
                                  bought_at,
                                  tickerInfo["ticker"],
                                  bought_price,
                                  stopLoss,
                                  reward,
                                  1, 
                                  "RSI",
                                  reason,
                                  False)
                
             
                 
        elif (actual_signal == "Buy"):
            
            profit = (stockPrice - bought_price)
            
            if high_profit == 0:
                high_profit = profit
            
            if (rsi_value >= 70):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index,
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price,
                                 profit,
                                 1, 
                                 "RSI", 
                                 "Above 70",
                                 False)
                   )
               
               
            #calculo o Stoploss
            if (stockPrice <= first_stopLoss):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 "RSI", 
                                 "Stop loss U$" + str(first_stopLoss),
                                 False)
                   )
               
               
            #calculo o novo reward
            if (stockPrice > actual_reward):
                actual_reward = reward
                if stopLoss > actual_stopLoss:
                    actual_stopLoss = stopLoss
    
            #calculo o novo reward
            if (profit > high_profit):
                actual_reward = reward
                high_profit = profit
                if stopLoss > actual_stopLoss:
                    actual_stopLoss = stopLoss
    
            if (profit > 0) and ((high_profit > 0) and (((profit*100) / high_profit) < safe_profit_reward_percent)) and (high_profit - profit > tickerInfo["reward_minimun_amount"]):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price,
                                 profit,
                                 1, 
                                 "RSI", 
                                 "Stop loss - "+str(safe_profit_reward_percent)+"% total earned"),
                                 False
                                 )
               
                
            
        result_signal.append(actual_signal)
        result_bought.append(bought_price)
        result_sold.append(sold_price)
        result_profit.append(profit)
        result_rsi_stop_loss.append(actual_stopLoss)
        result_close.append(stockPrice)
        result_rsi_reward.append(actual_reward)
        
        if (actual_signal == "Sell"):
           actual_signal = ""
           bought_at = ""
           bought_price = 0.00
           sold_price = 0.00
           profit = 0.00
           actual_stopLoss = 0.00
           high_profit = 0.00
           
           if (rsi_value < 30):
               below_value = True
        #RSI Buy/Sell Signal ======= END


    df_BuySell["RSI Signal"] = result_signal
    df_BuySell["RSI Bought"] = result_bought
    df_BuySell["RSI Sold"] = result_sold
    df_BuySell["RSI Close"] = result_close
    df_BuySell["RSI Profit"] = result_profit
    df_BuySell["RSI Stop Loss"] = result_rsi_stop_loss
    df_BuySell["RSI Reward"] = result_rsi_reward


    return df_BuySell, total_earned, result_stock_sold


def make_investment_calc_keltnerchannel(ohlcv, params):
    
    keltnerchannel_param = params["keltnerchannel"]
    userId = params["user_id"]
    tickerInfo = params["ticker"]
    window = keltnerchannel_param["window"]
    recovery_percent_after_down = keltnerchannel_param["recovery_percent_after_down"]
    
    df_BuySell = ohlcv.copy()
    df_BuySell = KeltnerChannel(df_BuySell, window)
    
    first_stopLoss = 0.00
    actual_stopLoss = 0.00
    actual_reward = 0.00
    actual_signal = ""
    high_profit = 0.00
    
    
    volume_greater = tickerInfo["volume_greater"]
    safe_profit_reward_percent = tickerInfo["reward_profit_saved_percent"]
    
    total_earned = 0.00
    
    result_signal = []
    result_bought = []
    result_sold = []
    result_close = []
    result_profit = []
    result_keltner_stop_loss = []
    result_keltner_reward = []
    
    result_stock_sold = []
    
    bought_at = ""
    bought_price = 0.00
    sold_price = 0.00
    profit = 0.00
    below_down_value = False
    above_up_value = False


    for index, row in df_BuySell.iterrows():
        stockPrice = round(row["Close"], 3)
        stopLoss = row["Stop Loss"]
        reward = row["Reward"]
        
        #Keltch Buy/Sell Signal ======= BEGIN
        kc_upper = row['Kelch_Upper']
        kc_middle = row['Kelch_Middle']
        kc_down = row['Kelch_Down']
        
        #Sem compra
        if (actual_signal == ""):
             #Monitoro para saber se bateu abaixo de 30
             if (stockPrice < kc_down):
                 below_down_value = True
                 
            
             #Voltou x% acima do down
             good_signal = True
             if (recovery_percent_after_down > 0 and (below_down_value == True)):
                 good_signal = (abs(((stockPrice*100)/kc_down)-100))*100 > recovery_percent_after_down
             
     
             #Voltou acima do down
             if ((below_down_value == True) and (good_signal == True) and (stockPrice > kc_down) and (row["Volume"] > volume_greater)):
                actual_signal = "Buy"
                bought_at = index
                bought_price = stockPrice
                sold_price = 0.00
                profit = 0.00
                actual_stopLoss = stopLoss
                first_stopLoss = stopLoss
                actual_reward = reward
                below_down_value = False
                
                reason = "Keltner Channel was bellow down AND Stock ({0}) > ({1}) AND Volume ({2}) > ({3})".format(stockPrice, kc_down, row["Volume"], volume_greater)
                register_purchase(userId,
                                bought_at,
                                tickerInfo["ticker"],
                                bought_price,
                                stopLoss,
                                reward,
                                1, 
                                "Keltner Channel",
                                reason,
                                False)
                
             
                 
        elif (actual_signal == "Buy"):
            
            profit = (stockPrice - bought_price)
            
            if high_profit == 0:
                high_profit = profit
                
            if (stockPrice >= kc_upper):
                above_up_value = True
            
            if (above_up_value == True) and (stockPrice < kc_middle):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index,
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price,
                                 profit,
                                 1, 
                                 "Keltner Channel", 
                                 "Avobe Upper and {0} < {1}".format(stockPrice, kc_middle)),
                                 False
                                 )
               
               
            #calculo o Stoploss
            if (stockPrice <= first_stopLoss):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 "Keltner Channel", 
                                 "Stop loss U$" + str(first_stopLoss),
                                 False)
                   )
               
               
            #calculo o novo reward
            if (stockPrice > actual_reward):
                actual_reward = reward
                if stopLoss > actual_stopLoss:
                    actual_stopLoss = stopLoss
    
            #calculo o novo reward
            if (profit > high_profit):
                actual_reward = reward
                high_profit = profit
                if stopLoss > actual_stopLoss:
                    actual_stopLoss = stopLoss
    
            if (profit > 0) and ((high_profit > 0) and (((profit*100) / high_profit) < safe_profit_reward_percent)) and (high_profit - profit > tickerInfo["reward_minimun_amount"]):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price,
                                 profit,
                                 1, 
                                 "Keltner Channel", 
                                 "Stop loss - "+str(safe_profit_reward_percent)+"% total earned",
                                 False)
                                 )
               
                
            
        result_signal.append(actual_signal)
        result_bought.append(bought_price)
        result_sold.append(sold_price)
        result_profit.append(profit)
        result_keltner_stop_loss.append(actual_stopLoss)
        result_close.append(stockPrice)
        result_keltner_reward.append(actual_reward)
        
        if (actual_signal == "Sell"):
           actual_signal = ""
           bought_at = ""
           bought_price = 0.00
           sold_price = 0.00
           profit = 0.00
           actual_stopLoss = 0.00
           high_profit = 0.00
           above_up_value = False
           below_down_value = False
        #RSI Buy/Sell Signal ======= END


    df_BuySell["Keltner Signal"] = result_signal
    df_BuySell["Keltner Bought"] = result_bought
    df_BuySell["Keltner Sold"] = result_sold
    df_BuySell["Keltner Close"] = result_close
    df_BuySell["Keltner Profit"] = result_profit
    df_BuySell["Keltner Stop Loss"] = result_keltner_stop_loss
    df_BuySell["Keltner Reward"] = result_keltner_reward


    return df_BuySell, total_earned, result_stock_sold


def make_investment_calc_mov_avg(ohlcv, params):
    
    mov_avg_param = params["movavg"]
    userId = params["user_id"]
    tickerInfo = params["ticker"]
    moving_avg = mov_avg_param["type"]
    short_window = mov_avg_param["short_window"]
    long_window = mov_avg_param["long_window"]
    
    volume_greater = tickerInfo["volume_greater"]
    safe_profit_reward_percent = tickerInfo["reward_profit_saved_percent"]
    
    df_BuySell = ohlcv.copy()
    df_BuySell = MovingAverage(df_BuySell, moving_avg, short_window, long_window)
    
    first_stopLoss = 0.00
    actual_stopLoss = 0.00
    actual_reward = 0.00
    actual_signal = ""
    high_profit = 0.00
    
    total_earned = 0.00
    
    result_signal = []
    result_bought = []
    result_sold = []
    result_close = []
    result_profit = []
    result_sma_stop_loss = []
    result_sma_reward = []
    result_sma_position = []
    
    result_stock_sold = []
    
    bought_at = ""
    bought_price = 0.00
    sold_price = 0.00
    profit = 0.00
    
    actual_position = 0

    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  

    for index, row in df_BuySell.iterrows():
        #calculando o meu risco
        stockPrice = round(row["Close"], 3)
        stopLoss = row["Stop Loss"]
        reward = row["Reward"]
        actual_position = 0
        

        #SMA Buy/Sell Signal ======= BEGIN
        short_value = row[short_window_col]
        long_value = row[long_window_col]
        #Sem compra
        if (actual_signal == ""):
     
             #Sinal de compra
             if (short_value > long_value) and (row["Volume"] > volume_greater):
                actual_signal = "Buy"
                bought_price = stockPrice
                bought_at = index
                sold_price = 0.00
                profit = 0.00
                actual_stopLoss = stopLoss
                first_stopLoss = stopLoss
                actual_reward = reward
                actual_position = 1
                
                reason = "{0} ({1}) > Signal ({2}) AND Volume ({3}) > ({4})".format(moving_avg, short_value, long_value, row["Volume"], volume_greater)
                register_purchase(userId,
                                bought_at,
                                tickerInfo["ticker"],
                                bought_price,
                                stopLoss,
                                reward,
                                1, 
                                moving_avg,
                                reason,
                                False)
             
                 
        elif (actual_signal == "Buy"):
            
            profit = (stockPrice - bought_price)
            
            if high_profit == 0:
                high_profit = profit
            
            #Sinal de venda, pode cair
            if (short_value < long_value):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               actual_position = -1
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index, 
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 moving_avg, 
                                 "Short < Long",
                                 False)
                                 )
               
               
               #result_stock_sold
               
            #calculo o Stoploss
            if (stockPrice <= first_stopLoss):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += profit
               actual_position = -1
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index,
                                 bought_at,
                                 tickerInfo["ticker"],
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 1, 
                                 moving_avg, 
                                 "Stop loss U$" + str(actual_stopLoss),
                                 False)
                                 )
               
               
            #calculo o novo reward
            if (stockPrice > actual_reward):
                actual_reward = reward
                actual_stopLoss = stopLoss
    
            #calculo o novo reward
            if (profit > high_profit):
                actual_reward = reward
                actual_stopLoss = stopLoss
                high_profit = profit
    
            #Garanto um lucro quando cair abaixo de 50% do que eu já ganhei
            #if (profit > 0) and ((high_profit > 0) and (((profit*100) / high_profit) < safe_profit_reward_percent)) and (high_profit - profit > .10):
            #   actual_signal = "Sell"
            #   sold_price = stockPrice
            #   total_earned += profit
            #   actual_position = -1
                    
                
        result_signal.append(actual_signal)
        result_bought.append(bought_price)
        result_sold.append(sold_price)
        result_profit.append(profit)
        result_sma_stop_loss.append(actual_stopLoss)
        result_close.append(stockPrice)
        result_sma_reward.append(actual_reward)
        result_sma_position.append(actual_position)
        
        if (actual_signal == "Sell"):
           actual_signal = ""
           bought_at = ""
           bought_price = 0.00
           sold_price = 0.00
           profit = 0.00
           actual_stopLoss = 0.00
           high_profit = 0.00
           actual_position = 0
        #SMA Buy/Sell Signal ======= END


    df_BuySell["Avg Position"] = result_sma_position
    df_BuySell[moving_avg + " Signal"] = result_signal
    df_BuySell[moving_avg + " Bought"] = result_bought
    df_BuySell[moving_avg + " Sold"] = result_sold
    df_BuySell[moving_avg + " Close"] = result_close
    df_BuySell[moving_avg + " Profit"] = result_profit
    df_BuySell[moving_avg + " Stop Loss"] = result_sma_stop_loss
    df_BuySell[moving_avg + " Reward"] = result_sma_reward


    return df_BuySell, total_earned, result_stock_sold


def make_investment_calc_strategy_1(ohlcv, params):
    
    strategy1 = params["strategy1"]
    

    #MovAvg    
    mov_avg_param = strategy1["movavg"]
    moving_avg = mov_avg_param["type"]
    moving_avg_apply = mov_avg_param["apply"]
    short_window = mov_avg_param["short_window"]
    long_window = mov_avg_param["long_window"]
    
    df_BuySell = ohlcv.copy()
    df_BuySell = MovingAverage(df_BuySell, moving_avg, short_window, long_window)
    
    #MACD
    macd_param = strategy1["macd"]
    macd_apply = macd_param["apply"]
    df_BuySell = MACD(df_BuySell, macd_param["fast_length"], macd_param["slow_length"], macd_param["macd_length"])
    
    #RSI
    rsi_param = strategy1["rsi"]
    rsi_line_bellow = rsi_param["line_bellow"]
    rsi_lengh = rsi_param["lengh"]
    rsi_enable_above_30 = rsi_param["enable_above_30"]
    rsi_apply = rsi_param["apply"]
    df_BuySell = RSI(df_BuySell, rsi_lengh)
    
    #VWAP
    vwap_param = strategy1["vwap"]
    vwap_apply = vwap_param["apply"]
    vwap_multiple = vwap_param["multiple"]
    df_BuySell = VWAP(df_BuySell, vwap_multiple)
    
    
    #Keltner Channel
    keltner_param = strategy1["keltnerchannel"]
    keltner_apply = keltner_param["apply"]
    keltner_recovery_percent_after_down = keltner_param["recovery_percent_after_down"]
    df_BuySell = KeltnerChannel(df_BuySell, keltner_param["window"])
    sell_when_bellow_down_line = keltner_param["sell_when_bellow_down_line"]
    sell_when_above_upper_line = keltner_param["sell_when_above_upper_line"]
    
    #TickerInfo
    tickerInfo = params["ticker"]
    volume_greater = tickerInfo["volume_greater"]
    safe_profit_reward_percent = tickerInfo["reward_profit_saved_percent"]
    minimun_reward_amount = tickerInfo["reward_minimun_amount"]
    ticker_code = tickerInfo["ticker"]
    stop_buy_after = tickerInfo["stop_buying_after_hour"]

    day_trade_mode = params["day_trade_mode"]
    
    userId = params["user_id"]
    total_amount_to_invest = params["invest_amount"]["total_amount"]
    ticker_amount_to_invest = params["invest_amount"]["ticker_amount"]
    qty_stock_bought = 1
    
    first_stopLoss = 0.00
    actual_stopLoss = 0.00
    actual_reward = 0.00
    actual_signal = ""
    high_profit = 0.00
    high_price = 0.00
    
    total_earned = 0.00
    
    result_signal = []
    result_bought = []
    result_sold = []
    result_close = []
    result_profit = []
    result_sma_stop_loss = []
    result_sma_reward = []
    result_sma_position = []
    
    result_stock_sold = []
    
    bought_at = ""
    bought_price = 0.00
    sold_price = 0.00
    profit = 0.00
    
    actual_position = 0
    rsi_was_bellow_30 = False
    
    vwap_bellow_support = False
    
    keltner_below_down_value = False
    keltner_above_up_value = False
    
    loop_date = ""

    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  

    for index, row in df_BuySell.iterrows():
        #calculando o meu risco
        stockPrice = round(row["Close"], 2)
        stopLoss = round(row["Stop Loss"], 2)
        reward = round(row["Reward"], 2)
        actual_position = 0
        

        #Buy/Sell Signal ======= BEGIN
        moving_avg_short_value = row[short_window_col]
        moving_avg_long_value = row[long_window_col]
        
        macd_value = row["MACD"]
        macd_signal = row["MA_Signal"]
        
        rsi_value = row["RSI"]
        
        volume_value = row["Volume"]
        
        vwap_value = row["VWAP"]
        vwap_support = row["VWAP_DEV_DN"]
        
        kc_upper = row['Kelch_Upper']
        kc_middle = row['Kelch_Middle']
        kc_down = row['Kelch_Down']
        
        #Sem compra
        if (actual_signal == ""):
     
             if (rsi_value < 30) or (rsi_enable_above_30 == False):
                 rsi_was_bellow_30 = True
                 
             if (stockPrice < vwap_support):
                 vwap_bellow_support = True
                 
             #Monitoro para saber se bateu abaixo de 30
             if (stockPrice < kc_down):
                 keltner_below_down_value = True

             if keltner_below_down_value and (stockPrice > kc_middle):
                 keltner_below_down_value = False
                 
             if (stockPrice > kc_upper):
                 keltner_above_up_value = True
            
             #Voltou x% acima do down
             keltner_good_signal = True
             if (keltner_recovery_percent_after_down > 0 and (keltner_below_down_value == True)):
                 keltner_good_signal = (abs(((stockPrice*100)/kc_down)-100))*100 > keltner_recovery_percent_after_down
                 
             
             #Mudou o dia, tenho que zerar tudo
             if (loop_date == "") or (loop_date != index.replace(tzinfo=None).date()):
                 rsi_was_bellow_30 = False
                 keltner_below_down_value = False
                 keltner_above_up_value = False
                 keltner_good_signal = False
                 macd_signal = 0
    
             
             #Sinal de compra
             volume_ok = (volume_value > volume_greater)
             macd_ok = (macd_signal < 0) and (macd_value > macd_signal) if macd_apply == True else True
             movavg_ok = (stockPrice > moving_avg_short_value) if moving_avg_apply == True else True
             rsi_ok = ((rsi_value < rsi_line_bellow) and (rsi_value > 0) and (rsi_was_bellow_30 == True)) if rsi_apply == True else True
             vwap_ok = (stockPrice > vwap_value) and (vwap_bellow_support == True) if vwap_apply == True else True
             keltner_ok = ((keltner_below_down_value == True) and (keltner_good_signal == True) and (stockPrice > kc_down)) if keltner_apply == True else True
             
             
             buy_now = (volume_ok and macd_ok and movavg_ok and rsi_ok and vwap_ok and keltner_ok)
        
             #verifico que é no mesmo dia para comprar
             if day_trade_mode:
                 if buy_now and (index.replace(tzinfo=None).date() != dt.datetime.now().date()):
                    buy_now = False
             
             #verifico se o horário é o ultimo do dia para não comprar mais
             if index.replace(tzinfo=None).hour >= stop_buy_after:
                 buy_now = False
        
             if (buy_now == True):
                actual_signal = "Buy"
                bought_at = index.replace(tzinfo=None)
                bought_price = stockPrice
                sold_price = 0.00
                profit = 0.00
                actual_stopLoss = stopLoss
                first_stopLoss = stopLoss
                actual_reward = reward
                actual_position = 1
                rsi_was_bellow_30 = False
                vwap_bellow_support = False
                
                reason = "Volume ({0}) > ({1})".format(row["Volume"], volume_greater)
                
                if macd_apply:
                    reason += " AND MACD Signal < 0 AND MACD({0}) > Signal({1})".format(macd_value, macd_signal)

                if moving_avg_apply:
                    reason += " AND {0} Stock Price({1}) > {2}".format(moving_avg, stockPrice, moving_avg_short_value)
                
                if rsi_apply:
                    reason += " AND RSI ({0} < {1})".format(rsi_value, rsi_line_bellow)
                    
                if vwap_apply:
                    reason += " AND VWAP ({0}) > Stock ({1})".format(vwap_value, stockPrice)
                
                if keltner_apply:
                    reason += " AND Keltner was < {0} down AND Stock {1} > {2}".format(kc_down, stockPrice, kc_down)
                
                #calculate the stock quantity to buy
                qty_stock_bought = math.ceil(ticker_amount_to_invest/bought_price)
                
                register_purchase(userId,
                                bought_at,
                                ticker_code,
                                bought_price,
                                stopLoss,
                                reward,
                                qty_stock_bought, 
                                strategy1["description"],
                                reason,
                                True)
                 
        elif (actual_signal == "Buy"):
            
            profit = (stockPrice - bought_price)
            
            qty_stock_bought = math.ceil(ticker_amount_to_invest/bought_price)
            
            if high_profit == 0:
                high_profit = profit
            
            if high_price < stockPrice:
                high_price = stockPrice
                
            #calculo o novo reward
            if (stockPrice > actual_reward):
                actual_reward = reward
                actual_stopLoss = stopLoss
    
            #calculo o novo reward
            if (profit > high_profit):
                actual_reward = reward
                actual_stopLoss = stopLoss
                high_profit = profit
                
            
            #calculo o Stoploss da largada
            if (stockPrice <= first_stopLoss):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += round(profit * qty_stock_bought, 2)
               actual_position = -1
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index,
                                 bought_at,
                                 ticker_code,
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 qty_stock_bought, 
                                 strategy1["description"], 
                                 "Stop loss U$" + str(round(first_stopLoss, 2)),
                                 True)
                                 )
               
            
            if (actual_signal == "Buy"):
                
                sell_now = False
                
                #if (stockPrice < high_price) and (profit > .10):
                if (profit > minimun_reward_amount):
                    
                    reason = "Profit > ${0} ".format(minimun_reward_amount)
                    
                    percent_loss = round(((high_price-stockPrice)/high_price)*100, 2)
                    
                    sell_now = (percent_loss < safe_profit_reward_percent)
                    if sell_now:
                        reason += " AND %Loss {0} < %Reward {1} ".format(percent_loss, safe_profit_reward_percent)
                    
                    #trabalhar com o sinal da venda acompanhando o crescimento
                    #trabalhar em uma forma de esticar mais ganho (Esse não funcionou)
                    if keltner_apply and sell_when_bellow_down_line:
                       sell_now = (stockPrice < kc_down)

                    # testar para maximizar o ganho
                    if keltner_apply and sell_when_above_upper_line:
                       sell_now = (stockPrice > kc_upper)
                       reason += " AND Stock Price {0} > Keltner Upper {1} ".format(stockPrice, kc_upper)
                    
                    
                    if sell_now:
                        actual_signal = "Sell"
                        sold_price = stockPrice
                        total_earned += round(profit * qty_stock_bought, 2)
                        actual_position = -1
                        
                        result_stock_sold.append(
                            register_sale(userId,
                                          index, 
                                          bought_at,
                                          ticker_code,
                                          bought_price, 
                                          sold_price, 
                                          profit,
                                          qty_stock_bought, 
                                          strategy1["description"], 
                                          reason,
                                          True)
                                          )
            
               
        result_signal.append(actual_signal)
        result_bought.append(bought_price)
        result_sold.append(sold_price)
        result_profit.append(profit)
        result_sma_stop_loss.append(actual_stopLoss)
        result_close.append(stockPrice)
        result_sma_reward.append(actual_reward)
        result_sma_position.append(actual_position)
        
        loop_date = index.replace(tzinfo=None).date()
        
        if (actual_signal == "Sell"):
           actual_signal = ""
           bought_at = ""
           bought_price = 0.00
           sold_price = 0.00
           profit = 0.00
           actual_stopLoss = 0.00
           high_profit = 0.00
           actual_position = 0
           high_price = 0.00
           vwap_bellow_support = False
           keltner_above_up_value = False
           keltner_below_down_value = False
           rsi_was_bellow_30 = True if rsi_value < 30 else False
           qty_stock_bought = 1
           
        #Strategy 1 Buy/Sell Signal ======= END


    df_BuySell["Avg Position"] = result_sma_position
    df_BuySell["Stg Signal"] = result_signal
    df_BuySell["Stg Bought"] = result_bought
    df_BuySell["Stg Sold"] = result_sold
    df_BuySell["Stg Close"] = result_close
    df_BuySell["Stg Profit"] = result_profit
    df_BuySell["Stg Stop Loss"] = result_sma_stop_loss
    df_BuySell["Stg Reward"] = result_sma_reward


    #remove columns not used
    if macd_apply == False:
       df_BuySell = df_BuySell.drop(['MA_Fast', 'MA_Slow', 'MACD', 'MA_Signal'],axis=1)

    if moving_avg_apply == False:
       df_BuySell = df_BuySell.drop([short_window_col, long_window_col, 'Avg Position', 'Avg Signal'],axis=1)

    if vwap_apply == False:
       df_BuySell = df_BuySell.drop(['VWAP', 'VWAP_DEV_UP', 'VWAP_DEV_DN'],axis=1)

    if rsi_apply == False:
       df_BuySell = df_BuySell.drop(['RSI'],axis=1)
      
    if keltner_apply == False:
       df_BuySell = df_BuySell.drop(['Kelch_Upper', 'Kelch_Middle', 'Kelch_Down'],axis=1)
       
       
    return df_BuySell, total_earned, result_stock_sold

def make_investment_calc_strategy_2(ohlcv, params):
    
    strategy = params["strategy2"]
    
    #MovAvg    
    mov_avg_param = strategy["movavg"]
    moving_avg = mov_avg_param["type"]
    moving_avg_apply = mov_avg_param["apply"]
    window_1 = mov_avg_param["window_1"]
    window_2 = mov_avg_param["window_2"]
    window_3 = mov_avg_param["window_3"]
    window_4 = mov_avg_param["window_4"]
    
    df_BuySell = ohlcv.copy()
    #Mov 3 & 4
    df_BuySell = MovingAverage(df_BuySell, moving_avg, window_3, window_4)
    #Mov 1 & 2
    df_BuySell = MovingAverage(df_BuySell, moving_avg, window_1, window_2)
    
    #MACD
    macd_param = strategy["macd"]
    macd_apply = macd_param["apply"]
    df_BuySell = MACD(df_BuySell, macd_param["fast_length"], macd_param["slow_length"], macd_param["macd_length"])
    
   #VWAP
    vwap_param = strategy["vwap"]
    vwap_apply = vwap_param["apply"]
    vwap_multiple = vwap_param["multiple"]
    df_BuySell = VWAP(df_BuySell, vwap_multiple)
    
    #TickerInfo
    tickerInfo = params["ticker"]
    volume_greater = tickerInfo["volume_greater"]
    safe_profit_reward_percent = tickerInfo["reward_profit_saved_percent"]
    minimun_reward_amount = tickerInfo["reward_minimun_amount"]
    ticker_code = tickerInfo["ticker"]
    stop_buy_after = tickerInfo["stop_buying_after_hour"]

    day_trade_mode = params["day_trade_mode"]
    
    userId = params["user_id"]
    total_amount_to_invest = params["invest_amount"]["total_amount"]
    ticker_amount_to_invest = params["invest_amount"]["ticker_amount"]
    qty_stock_bought = 1
    
    first_stopLoss = 0.00
    actual_stopLoss = 0.00
    actual_reward = 0.00
    actual_signal = ""
    high_profit = 0.00
    high_price = 0.00
    
    total_earned = 0.00
    
    result_signal = []
    result_bought = []
    result_sold = []
    result_close = []
    result_profit = []
    result_sma_stop_loss = []
    result_sma_reward = []
    result_sma_position = []
    
    result_stock_sold = []
    
    bought_at = ""
    bought_price = 0.00
    sold_price = 0.00
    profit = 0.00
    
    actual_position = 0
    rsi_was_bellow_30 = False
    
    vwap_bellow_support = False
    
    macd_ready = False
    
    loop_date = ""

    window1_col = str(window_1) + '_' + moving_avg
    window2_col = str(window_2) + '_' + moving_avg  
    window3_col = str(window_3) + '_' + moving_avg
    window4_col = str(window_4) + '_' + moving_avg  

    for index, row in df_BuySell.iterrows():
        #calculando o meu risco
        stockPrice = round(row["Close"], 2)
        stopLoss = round(row["Stop Loss"], 2)
        reward = round(row["Reward"], 2)
        actual_position = 0

        #Buy/Sell Signal ======= BEGIN
        moving_avg_1_value = round(row[window1_col], 2)
        moving_avg_2_value =  round(row[window2_col], 2)
        moving_avg_3_value =  round(row[window3_col], 2)
        moving_avg_4_value =  round(row[window4_col], 2)
        
        macd_value = row["MACD"]
        macd_signal = row["MA_Signal"]
        
        volume_value = row["Volume"]
        
        vwap_value = row["VWAP"]
        vwap_support = row["VWAP_DEV_DN"]
        
        #Sem compra
        if (actual_signal == ""):
     
             if (stockPrice < vwap_support):
                 vwap_bellow_support = True
             
             #Mudou o dia, tenho que zerar tudo
             if (loop_date == "") or (loop_date != index.replace(tzinfo=None).date()):
                 macd_signal = 0
    
             if macd_ready == False:
                 macd_ready = (macd_signal < 0) and (macd_value > macd_signal)
    
             #Sinal de compra
             volume_ok = (volume_value > volume_greater)
             macd_ok = macd_ready if macd_apply == True else True
             movavg_ok = (stockPrice > moving_avg_1_value) and (stockPrice > moving_avg_2_value) and (stockPrice > moving_avg_3_value) if moving_avg_apply == True else True
             vwap_ok = (stockPrice > vwap_value) and (vwap_bellow_support == True) if vwap_apply == True else True
             
             buy_now = (volume_ok and macd_ok and movavg_ok and vwap_ok)
        
             #verifico que é no mesmo dia para comprar
             if day_trade_mode:
                 if buy_now and (index.replace(tzinfo=None).date() != dt.datetime.now().date()):
                    buy_now = False
             
             #verifico se o horário é o ultimo do dia para não comprar mais
             if index.replace(tzinfo=None).hour >= stop_buy_after:
                 buy_now = False
        
             macd_ready = (macd_signal > 0) and (macd_value < macd_signal)
        
             if (buy_now == True):
                actual_signal = "Buy"
                bought_at = index.replace(tzinfo=None)
                bought_price = stockPrice
                sold_price = 0.00
                profit = 0.00
                actual_stopLoss = stopLoss
                first_stopLoss = stopLoss
                actual_reward = reward
                actual_position = 1
                vwap_bellow_support = False
                macd_ready = False
                
                reason = "Volume ({0}) > ({1})".format(row["Volume"], volume_greater)
                
                if macd_apply:
                    reason += " AND MACD Signal < 0 AND MACD({0}) > Signal({1})".format(macd_value, macd_signal)

                if moving_avg_apply:
                    reason += " AND {0} > {1} {2} {3} {4}".format(stockPrice, moving_avg, moving_avg_1_value, moving_avg_2_value, moving_avg_3_value)
                
                if vwap_apply:
                    reason += " AND VWAP ({0}) > Stock ({1})".format(vwap_value, stockPrice)
                
                #calculate the stock quantity to buy
                qty_stock_bought = math.ceil(ticker_amount_to_invest/bought_price)
                
                if (bought_at.hour > 9) and (bought_at.hour < 12):
                    register_oportunity(userId,
                                    bought_at,
                                    ticker_code,
                                    bought_price,
                                    stopLoss,
                                    reward,
                                    qty_stock_bought, 
                                    strategy["description"],
                                    reason)
                 
        elif (actual_signal == "Buy"):
            
            profit = (stockPrice - bought_price)
            
            qty_stock_bought = math.ceil(ticker_amount_to_invest/bought_price)
            
            if high_profit == 0:
                high_profit = profit
            
            if high_price < stockPrice:
                high_price = stockPrice
                
            #calculo o novo reward
            if (stockPrice > actual_reward):
                actual_reward = reward
                actual_stopLoss = stopLoss
    
            #calculo o novo reward
            if (profit > high_profit):
                actual_reward = reward
                actual_stopLoss = stopLoss
                high_profit = profit
                
            
            #calculo o Stoploss da largada
            if (stockPrice <= first_stopLoss):
               actual_signal = "Sell"
               sold_price = stockPrice
               total_earned += round(profit * qty_stock_bought, 2)
               actual_position = -1
               
               result_stock_sold.append(
                   register_sale(userId,
                                 index,
                                 bought_at,
                                 ticker_code,
                                 bought_price, 
                                 sold_price, 
                                 profit,
                                 qty_stock_bought, 
                                 strategy["description"], 
                                 "Stop loss U$" + str(round(first_stopLoss, 2)),
                                 False)
                                 )
            
            if (actual_signal == "Buy"):
                
                sell_now = False
                
                #if (stockPrice < high_price) and (profit > .10):
                if (profit > minimun_reward_amount):
                    
                    reason = "Profit > ${0} ".format(minimun_reward_amount)
                    
                    percent_loss = round(((high_price-stockPrice)/high_price)*100, 2)
                    
                    sell_now = (percent_loss < safe_profit_reward_percent)
                    if sell_now:
                        reason += " AND %Loss {0} < %Reward {1} ".format(percent_loss, safe_profit_reward_percent)
                    
                    # reason maximaze
                    
                    sell_now = (row['Avg Position'] == -1)

                    
                    if sell_now:
                        actual_signal = "Sell"
                        sold_price = stockPrice
                        total_earned += round(profit * qty_stock_bought, 2)
                        actual_position = -1
                        
                        result_stock_sold.append(
                            register_sale(userId,
                                          index, 
                                          bought_at,
                                          ticker_code,
                                          bought_price, 
                                          sold_price, 
                                          profit,
                                          qty_stock_bought, 
                                          strategy["description"], 
                                          reason,
                                          False)
                                          )
               
        result_signal.append(actual_signal)
        result_bought.append(bought_price)
        result_sold.append(sold_price)
        result_profit.append(profit)
        result_sma_stop_loss.append(actual_stopLoss)
        result_close.append(stockPrice)
        result_sma_reward.append(actual_reward)
        result_sma_position.append(actual_position)
        
        loop_date = index.replace(tzinfo=None).date()
        
        if (actual_signal == "Sell"):
           actual_signal = ""
           bought_at = ""
           bought_price = 0.00
           sold_price = 0.00
           profit = 0.00
           actual_stopLoss = 0.00
           high_profit = 0.00
           actual_position = 0
           high_price = 0.00
           vwap_bellow_support = False
           qty_stock_bought = 1
           
        #Strategy 1 Buy/Sell Signal ======= END


    df_BuySell["Avg Position"] = result_sma_position
    df_BuySell["Stg Signal"] = result_signal
    df_BuySell["Stg Bought"] = result_bought
    df_BuySell["Stg Sold"] = result_sold
    df_BuySell["Stg Close"] = result_close
    df_BuySell["Stg Profit"] = result_profit
    df_BuySell["Stg Stop Loss"] = result_sma_stop_loss
    df_BuySell["Stg Reward"] = result_sma_reward


    #remove columns not used
    if macd_apply == False:
       df_BuySell = df_BuySell.drop(['MA_Fast', 'MA_Slow', 'MACD', 'MA_Signal'],axis=1)

    if moving_avg_apply == False:
       df_BuySell = df_BuySell.drop([window1_col, window2_col, window3_col, window4_col, 'Avg Position', 'Avg Signal'],axis=1)

    if vwap_apply == False:
       df_BuySell = df_BuySell.drop(['VWAP', 'VWAP_DEV_UP', 'VWAP_DEV_DN'],axis=1)

       
    return df_BuySell, total_earned, result_stock_sold

def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
        
    #diff_df = diff_df.drop(['_merge'],axis=1)
    
    return diff_df

def make_investment_calc_earnings(ohlcv, tickerInfo, topSecPerf):
    
    result = {}
    approved = False
    
    #MovAvg    
    moving_avg = "SMA"
    window_1 = 10
    window_2 = 20
    window_3 = 50
    window_4 = 100
    
    df_BuySell = ohlcv.copy()
    #Mov 3 & 4
    df_BuySell = MovingAverage(df_BuySell, moving_avg, window_3, window_4)
    #Mov 1 & 2
    df_BuySell = MovingAverage(df_BuySell, moving_avg, window_1, window_2)
    
    atr_window = 14
    df_BuySell = ATR(df_BuySell, atr_window)

    adr_days = 20
    adr_value = ADR(df_BuySell, adr_days)

    window1_col = str(window_1) + '_' + moving_avg
    window2_col = str(window_2) + '_' + moving_avg  
    window3_col = str(window_3) + '_' + moving_avg
    window4_col = str(window_4) + '_' + moving_avg  

    row = df_BuySell.tail(1)
    
    stockPrice = round(row.iloc[0]["Close"], 2)
    stockOpen = round(row.iloc[0]["Open"], 2)
    moving_avg_1_value = round(row.iloc[0][window1_col], 2)
    moving_avg_2_value =  round(row.iloc[0][window2_col], 2)
    moving_avg_3_value =  round(row.iloc[0][window3_col], 2)
    moving_avg_4_value =  round(row.iloc[0][window4_col], 2)
    volume_value = row.iloc[0]["Volume"]
    
    moving_avg_satisfied = False
    volume_satisfied = False
    atr_satisfied = False
    adr_satisfied = False
    volume_minimun = 200000
    
    if stockPrice > moving_avg_1_value and stockPrice > moving_avg_2_value and stockPrice > moving_avg_3_value:
        if moving_avg_1_value > moving_avg_2_value and moving_avg_2_value > moving_avg_3_value:
            moving_avg_satisfied = True
        
    volume_satisfied = volume_value > volume_minimun
    
    
    atr_value = 0.00 if pd.isnull(row.iloc[0]["ATR"]) else round(row.iloc[0]["ATR"], 2) 
    atr_satisfied = (abs(stockPrice - stockOpen) < atr_value)
    
    adr_satisfied = adr_value > 3.5
    
    approved = moving_avg_satisfied and volume_satisfied and adr_satisfied 
    approved = (tickerInfo["sector"] in topSecPerf)
    
    result = {
        "ticker": tickerInfo["ticker"],
  		  "company": tickerInfo["company"],
        "sector": tickerInfo["sector"],
        "industry": tickerInfo["industry"],
        "event_date": tickerInfo["event_date"],
        "call_time": tickerInfo["call_time"],
        "eps_estimated": tickerInfo["eps_estimated"],
        "eps_reported": tickerInfo["eps_reported"],
        "stock_price": stockPrice,
        "mov_avg_satisfied": bool(moving_avg_satisfied),
        "mov_avg_10MA": moving_avg_1_value,
        "mov_avg_20MA": moving_avg_2_value,
        "mov_avg_50MA": moving_avg_3_value,
        "volume_satisfied": bool(volume_satisfied),
        "volume_value": volume_value,
        "volume_minimum": volume_minimun,
        "atr_satisfied": bool(atr_satisfied),
        "atr_value": atr_value,
        "adr_satisfied": bool(adr_satisfied),
        "adr_value": adr_value
        }           

    return result, approved

def make_investment(params):
    
    df_BuySell = {}
    start_day = params["date_start"]
    end_day = params["date_end"]
    data_period = params["data_period"] # fetch data by interval (including intraday if period < 60 days). valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    data_interval = params["data_interval"] # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    apply_rsi = params["rsi"]["apply"]
    apply_keltnerchannel = params["keltnerchannel"]["apply"]
    apply_macd = params["macd"]["apply"]
    apply_sma = params["movavg"]["apply"]
    apply_strategy1 = params["strategy1"]["apply"]
    apply_strategy2 = params["strategy2"]["apply"]
    stock_after_hour = params["stock_after_hour"]

    tickerInfo = params["ticker"]
    ticker = tickerInfo["ticker"]
    stock_stop_loss_percent =  tickerInfo["stop_loss_percent"] 
    stock_reward_percent =  tickerInfo["reward_target_percent"] 
    stock_volume_greater =  tickerInfo["volume_greater"]
    
    total_reward_rsi = 0.00
    total_reward_keltnerchannel = 0.00
    total_reward_macd = 0.00
    total_reward_sma = 0.00
    total_reward_strategy1 = 0.00
    total_reward_strategy2 = 0.00
    total_reward = 0.00
    
    result_funcion_reward = []
    
    result_stock_sold = []
    result_stock_sold_ticker = []
    function_stok_sold = []
    
    if data_period == "":
        ohlcv = yf.download(ticker, start=start_day, end=end_day, interval=data_interval, prepost=stock_after_hour)
    else:
        ohlcv = yf.download(ticker, period=data_period, interval=data_interval, prepost=stock_after_hour)
    
    df_BuySell = ohlcv.copy()

    #calculo o StopLoss e Reward (Meu Risco)
    result_stop_loss = []
    result_reward = []
    volume_greater = []
    for index, row in df_BuySell.iterrows():
        stockPrice = round(row["Close"], 3)
        stopLoss = investiment_stop_loss(stockPrice, stock_stop_loss_percent) #1% stop loss
        reward = investiment_reward(stockPrice, stock_reward_percent) #5% ganho
        result_stop_loss.append(stopLoss)
        result_reward.append(reward)
        volume_greater.append(stock_volume_greater)
    
    df_BuySell["Volume Minimum"] = volume_greater
    df_BuySell["Stop Loss"] = result_stop_loss
    df_BuySell["Reward"] = result_reward
    
    
    if apply_rsi:
       df_BuySell, total_reward_rsi, function_stok_sold = make_investment_calc_rsi(df_BuySell, params)
       total_reward += total_reward_rsi
       result_funcion_reward.append([ticker, "RSI", total_reward_rsi])
       result_stock_sold.append(function_stok_sold)
       
    if apply_keltnerchannel:
       df_BuySell, total_reward_keltnerchannel, function_stok_sold = make_investment_calc_keltnerchannel(df_BuySell, params)
       total_reward += total_reward_keltnerchannel
       result_funcion_reward.append([ticker, "keltnerChnel", total_reward_keltnerchannel])
       result_stock_sold.append(function_stok_sold)

    if apply_macd:
       df_BuySell, total_reward_macd, function_stok_sold = make_investment_calc_macd(df_BuySell, params)
       total_reward += total_reward_macd
       result_funcion_reward.append([ticker, "MACD", total_reward_macd])
       result_stock_sold.append(function_stok_sold)

    if apply_sma:
       df_BuySell, total_reward_sma, function_stok_sold = make_investment_calc_mov_avg(df_BuySell, params)
       total_reward += total_reward_sma
       result_funcion_reward.append([ticker, params["movavg"]["type"], total_reward_sma])
       result_stock_sold.append(function_stok_sold)
    
    if apply_strategy1:
       df_BuySell, total_reward_strategy1, function_stok_sold = make_investment_calc_strategy_1(df_BuySell, params)
       total_reward += total_reward_strategy1
       result_funcion_reward.append([ticker, "Strategy1", total_reward_strategy1])
       result_stock_sold.append(function_stok_sold)

    if apply_strategy2:
       df_BuySell, total_reward_strategy2, function_stok_sold = make_investment_calc_strategy_2(df_BuySell, params)
       total_reward += total_reward_strategy2
       result_funcion_reward.append([ticker, "Strategy2", total_reward_strategy2])
       result_stock_sold.append(function_stok_sold)
    
    if len(result_stock_sold) > 0:
        result_stock_sold_ticker.append({
            "ticker": ticker,
            "saleInfo": result_stock_sold
            })
    
    return df_BuySell, total_reward, result_funcion_reward, result_stock_sold_ticker


def get_ticker_fundamental(ticker):
    stock = finvizfinance(ticker)
    stock_fundament = stock.TickerFundament()
    return stock_fundament

def check_earnings(tickerInfo, topSecPerf):
    result = {}
    approved = False;
    ticker = tickerInfo["ticker"]
    data_period = "6mo" # fetch data by interval (including intraday if period < 60 days). valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    data_interval = "1d" # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    stock_after_hour = False
    ohlcv = yf.download(ticker, period=data_period, interval=data_interval, prepost=stock_after_hour)
    df_BuySell = ohlcv.copy()
    
    result, approved = make_investment_calc_earnings(df_BuySell, tickerInfo, topSecPerf)
    
    return result, approved

################################ Earnings ####################################  

def get_earning_calendar(days_prior, days_after):
    # setting the dates
    start_date = (dt.datetime.now().date() - timedelta(days=days_prior))
    end_date = (dt.datetime.now().date() + timedelta(days=days_after))
    
    # downloading the earnings calendar
    yec = YahooEarningsCalendar()
    earnings_list = yec.earnings_between(start_date, end_date)
    return earnings_list

def build_tickers_earning(df):
    tickers = []
    
    DF = df.copy()
    
    for index, row in DF.iterrows():
    
        startdatetimetype = ""    
        ticker = row["ticker"]
    
        if row["startdatetimetype"] == "BMO":
            startdatetimetype = "Before Market Open"
        elif row["startdatetimetype"] == "AMC":
            startdatetimetype = "After Market Close"
        else:
            startdatetimetype = row["startdatetimetype"]
            
        try:
            ticker_fundamental = get_ticker_fundamental(ticker)
        except:
            ticker_fundamental["Sector"] = "None"
            ticker_fundamental["Industry"] = "None"
        
    
        ticker = {
            		"ticker": ticker,
            		"company": row["companyshortname"],
                   "sector": ticker_fundamental["Sector"],
                   "industry": ticker_fundamental["Industry"],
                   "event_date": row["startdatetime"][:10],
                   "call_time": startdatetimetype,
                   "eps_estimated": 0.00 if pd.isnull(row["epsestimate"]) else row["epsestimate"],
                   "eps_reported": 0.00 if pd.isnull(row["epsactual"]) else row["epsactual"]
           		}
        
        tickers.append(ticker)
    
    return tickers

def create_earnings_calendar():
    
    weekno = dt.datetime.today().weekday()
    start_day = 0
    end_day = 2 if weekno == 5 else 1
    
    earnings_list = get_earning_calendar(start_day,end_day)
    earnings_df = pd.DataFrame(earnings_list)
    earnings_df.head()

    return build_tickers_earning(earnings_df)

################################ Sector info ####################################   

class Overview:
    """Overview
    Getting information from the finviz group overview page.
    """
    def __init__(self):
        """initiate module"""
        self.BASE_URL = 'https://finviz.com/groups.ashx?{group}&v=110'
        self.url = self.BASE_URL.format(group='g=sector')
        self._loadSetting()

    def _loadSetting(self):
        """load all the groups."""
        soup = webScrap(self.url)
        selects = soup.findAll('select')

        # group
        options = selects[0].findAll('option')
        key = [i.text for i in options]
        value = []
        for option in options:
            temp = option['value'].split('?')[1].split('&')
            if len(temp) == 4:
                temp = '&'.join(temp[:2])
            else:
                temp = temp[0]
            value.append(temp)
        self.group_dict = dict(zip(key, value))

        # order
        options = selects[1].findAll('option')
        key = [i.text for i in options]
        value = [i['value'].split('&')[-1] for i in options]
        self.order_dict = dict(zip(key, value))

    def getGroup(self):
        """Get groups.
        Returns:
            groups(list): all the available groups.
        """
        return list(self.group_dict.keys())

    def getOrders(self):
        """Get orders.
        Returns:
            orders(list): all the available orders.
        """
        return list(self.order_dict.keys())

    def ScreenerView(self, group='Sector', order='Name'):
        """Get screener table.
        Args:
            group(str): choice of group option.
            order(str): sort the table by the choice of order.
        Returns:
            df(pandas.DataFrame): group information table.
        """
        if group not in self.group_dict:
            raise ValueError()
        if order not in self.order_dict:
            raise ValueError()
        self.url = self.BASE_URL.format(group=self.group_dict[group])+'&'+self.order_dict[order]

        soup = webScrap(self.url)
        table = soup.findAll('table')[5]
        rows = table.findAll('tr')
        table_header = [i.text for i in rows[0].findAll('td')][1:]
        df = pd.DataFrame([], columns=table_header)
        rows = rows[1:]
        num_col_index = [i for i in range(2, len(table_header))]
        for row in rows:
            cols = row.findAll('td')[1:]
            info_dict = {}
            for i, col in enumerate(cols):
                # check if the col is number
                if i not in num_col_index:
                    info_dict[table_header[i]] = col.text
                else:
                    info_dict[table_header[i]] = numberCovert(col.text)

            df = df.append(info_dict, ignore_index=True)
        return df

class Performance(Overview):
    """Performance inherit from overview module.
    Getting information from the finviz group performance page.
    """
    def __init__(self):
        """initiate module
        """
        self.BASE_URL = 'https://finviz.com/groups.ashx?{group}&v=140'
        self.url = self.BASE_URL.format(group='g=sector')
        Overview._loadSetting(self)
        
def get_sector_performance():
    result = []
    top_results = []
    df_sectors = Performance().ScreenerView()    
    df_sectors = df_sectors.sort_values(by=['Perf Week'], ascending=False)
    loop = 0

    for index, row in df_sectors.iterrows():
        
        if loop < 3:
           top_results.append(row["Name"]) 
           
        loop += 1
        
        sector = {
            			"sector": row["Name"],
            			"perf_week": row["Perf Week"].replace("%", ""),
            			"perf_month": row["Perf Month"],
                   "perft_quart": row["Perf Quart"],
                   "perft_half": row["Perf Half"],
                   "perft_year": row["Perf Year"],
                   "change": row["Change"]
                   }
        
        result.append(sector)
    
    return result, top_results
        

################################ APIS ####################################    

def get_investor_info(userId):
    try:
        URL = "https://gcdekm9x7k.execute-api.us-west-1.amazonaws.com/default/investimentInfo?user_id={0}".format(userId)
        HEADERS = {"x-api-key": "ajWgX13WZ84rdArqDm7Jp1UjlUpSbsX3Jxyv2Cx"}
        result = requests.get(url=URL, headers=HEADERS)
        statusOK = result.status_code == 200
    except:
        result = None
        statusOK = False
    return result.json(), statusOK

def send_investor_transaction(body):
    try:
        URL = "https://ybj1f9tvq2.execute-api.us-west-1.amazonaws.com/default/transactionInvestment"
        HEADERS = {"Content-Type": "application/json", "x-api-key": "qFX0NOVe18laMdb2cpMX9OUzuDuSgWLa4WfmkE1"}
        result = requests.post(url=URL, headers=HEADERS, json=body)
        statusOK = result.status_code == 200
    except:
        result = None
        statusOK = False
        
    return result.json(), statusOK

def send_earnings(body):
    try:
        URL = "https://13h0fomfeb.execute-api.us-east-1.amazonaws.com/default/earningsAnalysis"
        HEADERS = {"Content-Type": "application/json", "x-api-key": "fGLiub6OCi5S7duaIBQEafNwCZHRUfM37gxnRnD"}
        result = requests.post(url=URL, headers=HEADERS, json=body)
        statusOK = result.status_code == 200
    except:
        result = None
        statusOK = False
        
    return result.json(), statusOK

def send_sector_performance(body):
    try:
        URL = "https://dzeityitjb.execute-api.us-east-1.amazonaws.com/default/sectorperformance"
        HEADERS = {"Content-Type": "application/json", "x-api-key": "GdMNlGxXU41jpV9Rq9jM5M1Mf6QIQ5l4JhWkvjY"}
        result = requests.post(url=URL, headers=HEADERS, json=body)
        statusOK = result.status_code == 200
    except:
        result = None
        statusOK = False
        
    return result.json(), statusOK

"""
body = {
	"userId": "131efbb2-83a0-4786-9241-1625c8cb801a",
	"boughtAt": "2021-03-18 09:30:00",
	"ticker": "AMZN",
	"study": "MACD + EAM + RSI + VWAP + KELCHN",
	"boughtPrice": 150.00,
	"stopLoss": 99.00,
	"reward": 180.00,
	"qty": 1,
	"buyReason": "Volume (408901.0) > (100000) AND MACD Signal < 0 AND MACD(-0.0567388493888501) > Signal(-0.06115664121084965) AND EMA Stock Price(24.49) > 24.398254998811193",
	"type": "bought"
    }

result, status = send_investor_transaction(body)
"""
################################ Test automation ####################################    

active_unit = False
active_robo = False
test_function = False
active_earning = True

df_tickerBought = pd.DataFrame(columns=["userid", "ticker", "boughtat", "bought"])


#SMA Strategy
#Buy shares of a stock when its 30-day moving average goes above the 100-day moving average.

#Sell shares of the stock when its 30-day moving average goes below the 100-day moving average.

#MongoDB 3a3daf5c-af75-44f1-9dde-15a6f7f30053


################################ Earning calculation ####################################
if active_earning:

    result = []
    tickers_calendar = create_earnings_calendar()
    sectorPerf, topSecPerf = get_sector_performance()

    for ticker in tickers_calendar:
        try:
            ticker_result, approved = check_earnings(ticker, topSecPerf) 
            
            if approved:
                result.append(ticker_result)
                
        except:
            print(ticker["ticker"]," Erro calculando valores")
    
    print("Top Industry", topSecPerf)
    print("Earning", result)
    reqresult, status = send_earnings(result)
    print(status, "Earning created")

    print("Sector performance", sectorPerf)
    reqresult, status = send_sector_performance(sectorPerf)
    print(status, "Sector performance created")
    

################################ Automation ####################################
if active_robo:
    
    userId = "131efbb2-83a0-4786-9241-1625c8cb801a"
    print("Loading API data for user {0}".format(userId))
    params, status = get_investor_info(userId)
    if status:
        print("API data loaded {0}".format(params["tickers"]))
    else:
        print("Error loading API")
    
    NotifyStartDayTrade(params["tickers"])
    
    df_BuySell = {}
    
    grand_total_reward = 0.00
    day_goal = params["invest_amount"]["today_goal_amount"]

    '''    
    params["tickers"] = [
            {
        			"ticker": "ACY",
        			"stop_loss_percent": 1,
        			"reward_target_percent": 3,
        			"reward_minimun_amount": 0.02,
        			"reward_profit_saved_percent": 50,
        			"stop_buying_after_hour": 15,
        			"volume_greater": 100000,
                "sector": "",
                "industry": ""
       		}
        ]
    
    
    '''
    params["strategy1"]["apply"] = False
    params["strategy2"]["apply"] = True
    day_goal = 500
    
    starttime=time.time()
    timeout = time.time() + 60*(60*10)  # horas
    keepInvesting = True
    while keepInvesting:
        try:
            investment_info = []
            investment_info.append(["Ticker", "U$"])
    
            investment_function_info = []
            investment_function_info.append(["Ticker", "Func", "U$"])
            
            investment_result_info = []
            investment_result_info.append(["Ticker", "Func", "Bought at", "Sold at", "Bought", "Sold", "Profit", "%", "Stocks", "Gain"])
    
            sale_info = []
    
            #clear console
            #print('\x1b[2J')
    
            keepInvesting = time.time() <= timeout
            grand_total_reward = 0.00
            tickers = params["tickers"]
            
            if len(tickers) == 0:
                print("No stocks to invest")
                keepInvesting = False
            
            for ticker in tickers:
                try:
                    ticker_code = ticker["ticker"]
                    params["ticker"] = ticker
                    
                    df_result, total_reward, result_per_funcion, result_sale_info = make_investment(params)
                    df_BuySell[ticker_code] = df_result
                    grand_total_reward += total_reward
                    
                    investment_info.append([ticker, total_reward])
                    investment_function_info.extend(result_per_funcion)
                    sale_info.append(result_sale_info)
                    
                    for sales in result_sale_info:
                        for item in sales['saleInfo']:
                            for func in item:
                                investment_result_info.append([ticker_code, 
                                                               func["study"], 
                                                               func["boughtAt"],
                                                               func["soldAt"],
                                                               func["boughtPrice"],
                                                               func["soldPrice"],
                                                               func["profit"],
                                                               str(round(((func["boughtPrice"]*100)/func["soldPrice"]-100)*-1,2)) + "%",
                                                               func["qty"],
                                                               str(round(func["qty"]*func["profit"],2))
                                                           ])
                except:
                    print(ticker_code," Erro calculando valores")
    
            print(tabulate(investment_result_info, headers="firstrow"))    
            print(tabulate(investment_function_info, headers="firstrow"))
            #print(tabulate(investment_info, headers="firstrow"))
            print("Grand total reward U$:", str(round(grand_total_reward, 2)))
            
            #print(sale_info)
            if grand_total_reward >= day_goal:
                print("GOALLLL!!!!!")
                keepInvesting = False
                
            if dt.datetime.now().hour > 20:
                print("Stop investing after {0}hs".format(17))
                keepInvesting = False
            
            if keepInvesting:
                time.sleep(60 * 1) # 30 segundos
        except KeyboardInterrupt:
            print('\n\nKeyboard exception received. Exiting.')
            exit()

################################ Individual ####################################        
if active_unit:

    #Keltner + MACD
    #volume_greater > 100K and sell_when_bellow_down_line: True (conservador)
    #volume_greater > 10K and sell_when_bellow_down_line: False (agressivo)

    params = {
        "user_id": "131efbb2-83a0-4786-9241-1625c8cb801a",
        "user_name": "Rodrigo Costa",
        "stock_after_hour": False,
        "data_period": "2d",
        "data_interval": "1m",
        "date_start": "2021-03-05 00:00:00.345727",
        "date_end": "2021-03-06 00:00:00.345727",
        "day_trade_mode": False,
        "invest_amount": {
            "initial_amount": 1000,
            "total_amount": 1500,
            "ticker_amount": 500,
            "today_goal_amount": 50
            },
        "rsi": {
            "apply": False,
            "recovery_above_30": 40
            },
        "keltnerchannel": {
            "apply": False,
            "window": 14,
            "recovery_percent_after_down": 6
            },
        "macd": {
            "apply": False,
            "fast_length": 12,
            "slow_length": 26,
            "macd_length": 9
            },
        "movavg": {
            "apply": False,
            "type": "EMA",
            "short_window": 5,
            "long_window": 75
            },
        "strategy1": {
            "apply": False,
            "description": "MACD + KELCHN",
            "macd": {
                "apply": True,
                "fast_length": 12,
                "slow_length": 26,
                "macd_length": 9
                },
            "movavg": {
                "apply": False,
                "type": "EMA",
                "short_window": 8,
                "long_window": 20
                },
            "vwap": {
                "apply": False,
                "multiple": 2.01
                },
            "rsi": {
                "apply": False,
                "lengh": 14,
                "line_bellow": 70,
                "enable_above_30": True
                },
            "keltnerchannel": {
                "apply": True,
                "window": 14,
                "recovery_percent_after_down": 6,
                "sell_when_bellow_down_line": False,
                "sell_when_above_upper_line": True
                }
            },
        "strategy2": {
            "apply": True,
            "description": "TAKE A LOOK. BIG CHANCE!",
            "macd": {
                "apply": False,
                "fast_length": 12,
                "slow_length": 26,
                "macd_length": 9
                },
            "movavg": {
                "apply": True,
                "type": "SMA",
                "window_1": 10,
                "window_2": 20,
                "window_3": 50,
                "window_4": 200
                },
            "vwap": {
                "apply": False,
                "multiple": 2.01
                }
            },
        "ticker": None,
        "tickers": [
            {
            "ticker": "AGC",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "4.3 %",
            "volume_greater": 100000,
            "sector": "Financial",
            "industry": "Shell Companies"
        },
        {
            "ticker": "AGCUU",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "5.75 %",
            "volume_greater": 100000,
            "sector": "Financial",
            "industry": "Shell Companies"
        },
        {
            "ticker": "BTBT",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "8.95 %",
            "volume_greater": 100000,
            "sector": "Technology",
            "industry": "Software - Application"
        },
        {
            "ticker": "IHRT",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "4.29 %",
            "volume_greater": 100000,
            "sector": "Communication Services",
            "industry": "Broadcasting"
        },
        {
            "ticker": "IZEA",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.05,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "4.5 %",
            "volume_greater": 100000,
            "sector": "Communication Services",
            "industry": "Internet Content & Information"
        },
        {
            "ticker": "LGVN",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.05,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "31.65 %",
            "volume_greater": 100000,
            "sector": "Healthcare",
            "industry": "Biotechnology"
        },
        {
            "ticker": "MFNC",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "49.58 %",
            "volume_greater": 100000,
            "sector": "Financial",
            "industry": "Banks - Regional"
        },
        {
            "ticker": "STAF",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.02,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "4.51 %",
            "volume_greater": 100000,
            "sector": "Industrials",
            "industry": "Staffing & Employment Services"
        },
        {
            "ticker": "VERU",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "5.12 %",
            "volume_greater": 100000,
            "sector": "Healthcare",
            "industry": "Drug Manufacturers - Specialty & Generic"
        },
        {
            "ticker": "VTVT",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.02,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "11.28 %",
            "volume_greater": 100000,
            "sector": "Healthcare",
            "industry": "Biotechnology"
        },
        {
            "ticker": "VUZI",
            "stop_loss_percent": 1,
            "reward_target_percent": 3,
            "reward_minimun_amount": 0.1,
            "reward_profit_saved_percent": 50,
            "stop_buying_after_hour": 15,
            "gap": "4.38 %",
            "volume_greater": 100000,
            "sector": "Technology",
            "industry": "Consumer Electronics"
        }                           
        ]
        }
    
    df_BuySell = {}
    
    investment_info = []
    investment_info.append(["Ticker", "U$"])

    investment_function_info = []
    investment_function_info.append(["Ticker", "Func", "U$"])
    investment_result_info = []
    investment_result_info.append(["Ticker", "Func", "Bought at", "Sold at", "Bought", "Sold", "Profit", "%", "Stocks", "Gain"])

    sale_info = []
    
    grand_total_reward = 0.00
    
    tickers = params["tickers"] if params["ticker"] == None else [params["ticker"]]
    for ticker in tickers:
        try:
            ticker_code = ticker["ticker"]
            params["ticker"] = ticker
            
            df_result, total_reward, result_per_funcion, result_sale_info = make_investment(params)
            df_BuySell[ticker_code] = df_result
            grand_total_reward += total_reward
            
            investment_info.append([ticker_code, total_reward])
            investment_function_info.extend(result_per_funcion)
            sale_info.append(result_sale_info)
            
            for sales in result_sale_info:
                for item in sales['saleInfo']:
                    for func in item:
                        investment_result_info.append([ticker_code, 
                                                       func["study"], 
                                                       func["boughtAt"],
                                                       func["soldAt"],
                                                       func["boughtPrice"],
                                                       func["soldPrice"],
                                                       func["profit"],
                                                       str(round(((func["boughtPrice"]*100)/func["soldPrice"]-100)*-1,2)) + "%",
                                                       func["qty"],
                                                       str(round(func["qty"]*func["profit"],2))
                                                   ])
            
            #print(ticker)
            #print(tabulate(result_per_funcion))
        except Exception as e:
            print(ticker_code,"Erro calculando valores", e)
    
    print(tabulate(investment_result_info, headers="firstrow"))    
    print(tabulate(investment_function_info, headers="firstrow"))
    #print(tabulate(investment_info, headers="firstrow"))
    print("Grand total reward U$:", str(round(grand_total_reward, 2)))


#ESTUDAR
#https://www.kaggle.com/kratisaxena/stock-market-technical-indicators-visualization


if test_function:
    
    ################################ Unit Teste ####################################
    # Download historical data for required stocks
    ticker = "MARA"
    #days = 120
    #ohlcv = yf.download(ticker, start=dt.date.today()-dt.timedelta(days), end=dt.datetime.today())
    ohlcv = yf.download(ticker, period="1d", interval="1m")
    
    
    ################################ RSI ####################################
    moving_avg = "EMA" #(str)the type of moving average to use ('SMA' or 'EMA')
    short_window = 10
    long_window = 20
    
    df_sma = MovingAverage(ohlcv, moving_avg, short_window, long_window)
    MovingAverageGraph(df_sma, ticker, moving_avg, short_window, long_window)
    MovingAveragePrint(df_sma)
    
    
    ################################ BollingerBand ####################################
    df_bollin = BollingerBand(ohlcv)
    BollingerBandGraph(df_bollin, ticker)
    
    ################################ MACD ####################################
    df_mdac = MACD(ohlcv, 12, 26, 9)
    MACDGraph(df_mdac, ticker)
    
    ################################ RSI ####################################
    df_rsi = RSI(ohlcv, 14)
    RSIGraph(df_rsi, ticker)
    
    ################################ VWAP ####################################
    df_vwap = VWAP(ohlcv, 2.01)
    VWAPGraph(df_vwap)
    
    ################################ Suppoert Registence ####################################
    levels = supportResistence(ohlcv)
    supportResistenceGraph(ohlcv, levels)

    ################################ Stochastic ####################################
    df_stok = Stochastic(ohlcv, 14)
    StochasticGraph(df_stok)
    
 
