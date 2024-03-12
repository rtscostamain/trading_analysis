import pandas as pd
import yfinance as yf
import matplotlib.dates as mpl_dates


#Average True Range  
def ATR(df, n):  
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



#Keltner Channel  
def KELCH(df, n):  
    """Calculate Keltner Channel for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
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



DF = yf.download("GOOGL", period="1d", interval="1m")
data1 = DF.copy()
data1['Date'] = pd.to_datetime(data1.index)
data1['Date'] = data1['Date'].apply(mpl_dates.date2num)

#read the data from the csv file
#data1 = pd.read_csv('E:/MarketData/EURUSD60.csv', header=None) 
#data1.columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
#data1.shape
#show data

data1=ATR(data1,14)
data1=KELCH(data1,14)
data1.tail()

data1.dropna(inplace=True)
data1.iloc[-100:,[-4,-3,-2]].plot(title="Keltner Channel")

#plot the Keltner Channels
import matplotlib.pyplot as plt
data1['Kelch_Middle'].plot(figsize=(12,8));
data1['Kelch_Upper'].plot(figsize=(12,8));
data1['Kelch_Down'].plot(figsize=(12,8));
plt.show()