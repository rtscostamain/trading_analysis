import pandas as pd
import mplfinance as mpf
import math
import yfinance as yf

'''
A VWAP não deve ser usada como um indicador de momentum, tais como MACD ou IFR.
Ela não dá sinais de sobre-compra ou sobre-venda, sendo este tipo de leitura totalmente inadequada para este indicador.

'''
#df = pd.read_csv('./data.csv', sep=',', quotechar='"')

df = yf.download("AMC", period="1d", interval="1m")

#df.set_index(['Date'], inplace=True)
#df.index = pd.to_datetime(df.index)
#df.index.name = 'Date'

# from here = https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/volume-weighted-average-price-vwap/
df['VWAP'] = (df.Volume * (df.High + df.Low) / 2).cumsum() / df.Volume.cumsum()
df['VWAP_MEAN_DIFF'] = ((df.High + df.Low) / 2) - df.VWAP
df['SQ_DIFF'] = df.VWAP_MEAN_DIFF.apply(lambda x: math.pow(x, 2))
df['SQ_DIFF_MEAN'] = df.SQ_DIFF.expanding().mean()
df['STDEV_TT'] = df.SQ_DIFF_MEAN.apply(math.sqrt)

stdev_multiple_1 = 1.28
stdev_multiple_2 = 2.01
stdev_multiple_3 = 2.51

df['STDEV_1'] = df.VWAP + stdev_multiple_1 * df['STDEV_TT']
df['STDEV_N1'] = df.VWAP - stdev_multiple_1 * df['STDEV_TT']

addplot  = [
    mpf.make_addplot(df['VWAP']),
    #mpf.make_addplot(df['STDEV_1']),
    #mpf.make_addplot(df['STDEV_N1']),
]

mpf.plot(df, type='candle', addplot=addplot)