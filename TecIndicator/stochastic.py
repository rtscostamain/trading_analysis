import pandas as pd
import mplfinance as mpf
import math
import yfinance as yf
import matplotlib.pyplot as plt

'''
A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time
'''
#df = pd.read_csv('./data.csv', sep=',', quotechar='"')

df = yf.download("TSLA", period="1d", interval="1m")

def Stochastic(df, n):
    df['STO_K'] = ((df['Close'] - df['Low'].rolling(window=n, center=False).mean()) / (df['High'].rolling(window=n, center=False).max() - df['Low'].rolling(window=n, center=False).min())) * 100
    df['STO_D'] = df['STO_K'].rolling(window = 3, center=False).mean()
    df["STO_K"] = df["STO_K"].fillna(0)
    df["STO_D"] = df["STO_D"].fillna(0)
    return df

df_stok = Stochastic(df, 14)


ax = df[['STO_K', 'STO_D']].plot()
df['Adj Close'].plot(ax=ax, secondary_y=True)
ax.axhline(20, linestyle='--', color="r")
ax.axhline(80, linestyle="--", color="r")
plt.show()

#https://tradingstrategyguides.com/best-stochastic-trading-strategy/
#Step #1: Check the daily chart and make sure the Stochastic indicator is below the 20 line and the %K line crossed above the %D line.
#Step #2: Move Down to the 15-Minute Time Frame and Wait for the Stochastic Indicator to hit the 20 level. The %K line(blue line) crossed above the %D line(orange line).
#Step #3: Wait for the Stochastic %K line (blue moving average) to cross above the 20 level
#Step #4: Wait for a Swing Low Pattern to develop on the 15-Minute Chart
#Step #5: Entry Long When the Highest Point of the Swing Low Pattern is Broken to the Upside

