
# Import necesary libraries
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# Download historical data for required stocks
ticker = "STNE"
ohlcv = yf.download(ticker, dt.date.today()-dt.timedelta(150), dt.datetime.today())


def BollingerBand(DF):
    """function to calculate Bollinger band
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA20"] = df["Adj Close"].rolling(window=20).mean()
    df["20dSTD"] = df["Adj Close"].rolling(window=20).std()
    
    df["Upper"] = df['MA20'] + (df['20dSTD'] * 2)
    df["Lower"] = df['MA20'] - (df['20dSTD'] * 2)
    df.dropna(inplace=True)
    return df


df = BollingerBand(ohlcv)

df[['Adj Close','MA20','Upper','Lower']].plot(figsize=(10,4))
plt.grid(True)
plt.title(ticker + ' Bollinger Bands')
plt.axis('tight')
plt.ylabel('Price')

# Closing prices above the upper Bollinger band may indicate that currently the stock price is too high and price may decrease soon.
# Closing prices below the lower Bollinger band may be seen as a sign that prices are too low and they may be moving up soon.