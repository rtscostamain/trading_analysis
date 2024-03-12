# =============================================================================
# FXCM API testing
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================
import fxcmpy

#initiating API connection and defining trade parameters
con = fxcmpy.fxcmpy(access_token="xxxx", log_level='error', server='demo')
pair = 'EUR/USD'


#get historical data
data = con.get_candles(pair, period='m5', number=250)
"""periods can be m1, m5, m15 and m30, H1, H2, H3, H4, H6 and H8, D1, W1, M1"""

#streaming data
"for streaming data, we first need ti subscribe to a currency pair"
currency_pair = "ETH/USD" #'EUR/USD'
con.subscribe_market_data(currency_pair)
con.get_last_price(currency_pair)
con.get_prices(currency_pair)
con.unsubscribe_market_data(currency_pair)

#trading account data
con.get_accounts().T

con.get_open_positions().T
con.get_open_positions_summary().T

con.get_closed_positions()

con.get_orders() 

#orders
con.create_market_buy_order(currency_pair, 10)
con.create_market_buy_order('USD/CAD', 10)
con.create_market_sell_order('USD/CAD', 20)
con.create_market_sell_order(currency_pair, 10)

order = con.open_trade(symbol='USD/CAD', is_buy=False,
                       is_in_pips=True,
                       amount=10, time_in_force='GTC',
                       stop=-9, trailing_step =True,
                       order_type='AtMarket', limit=9)

#con.close_trade(trade_id=tradeId, amount=1000)
con.close_all_for_symbol('USD/CAD')

#closing connection
con.close()