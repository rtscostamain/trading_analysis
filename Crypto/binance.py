from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd

api_key = "xxx"
api_secret = "xxx"

client = Client(api_key, api_secret)

depth = client.get_order_book(symbol='BNBBTC')

# get all symbol prices
prices = client.get_all_tickers()
df_skel = list()

for item in prices:
    df_skel.append(item)

df_prices = pd.DataFrame(df_skel)


klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1DAY, "90 days ago UTC")

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

# fetch 30 minute klines for the last month of 2017
klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# fetch weekly klines since it listed
klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")

