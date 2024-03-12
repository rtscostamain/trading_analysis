# =============================================================================
# Import OHLCV data and calculate ATR and Bollinger Band
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import yfinance as yf
import datetime as dt

# Download historical data for required stocks
ticker = "HTZGQ"
ohlcv = yf.download(ticker, period="6mo", interval="1d")


def ATR(DF,n):
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


def BollBnd(DF,n):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['Adj Close'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['Adj Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['Adj Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df.dropna(inplace=True)
    return df

def ADR(DF, n):
    "function to calculate Average Daily Range"
    df = ohlcv.copy()
    df['ADR_Div']=df['High']/df['Low']
    df_adr = df.tail(n)
    adr = 0.0
    index = 0
    
    for index, row in df_adr.iterrows():
        adr = adr + row['ADR_Div']
    
    adr_percent = round(100*((adr)/n-1),2)
    
    return adr_percent


# Visualizing Bollinger Band of the stocks for last 100 data points
#BollBnd(ohlcv,20).iloc[-100:,[-4,-3,-2]].plot(title="Bollinger Band")

adr = ADR(ohlcv, 20)
