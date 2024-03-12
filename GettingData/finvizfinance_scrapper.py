from finvizfinance.screener.overview import Overview
from finvizfinance.util import webScrap, progressBar, NUMBER_COL
import pandas as pd
import requests

columns = {
    0: 'No.',
    1: 'Ticker',
    2: 'Company',
    3: 'Sector',
    4: 'Industry',
    5: 'Country',
    6: 'Market Cap.',
    7: 'P/E',
    8: 'Forward P/E',
    9: 'PEG',
    10: 'P/S',
    11: 'P/B',
    12: 'P/Cash',
    13: 'P/Free Cash Flow',
    14: 'Dividend Yield',
    15: 'Payout Ratio',
    16: 'EPS',
    17: 'EPS growth this year',
    18: 'EPS growth next year',
    19: 'EPS growth past 5 years',
    20: 'EPS growth next 5 years',
    21: 'Sales growth past 5 years',
    22: 'EPS growth qtr over qtr',
    23: 'Sales growth qtr over qtr',
    24: 'Shares Outstanding',
    25: 'Shares Float',
    26: 'Insider Ownership',
    27: 'Insider Transactions',
    28: 'Institutional Ownership',
    29: 'Institutional Transactions',
    30: 'Float Short',
    31: 'Short Ratio',
    32: 'Return on Assets',
    33: 'Return on Equity',
    34: 'Return on Investments',
    35: 'Current Ratio',
    36: 'Quick Ratio',
    37: 'Long Term Debt/Equity',
    38: 'Total Debt/Equity',
    39: 'Gross Margin',
    40: 'Operating Margin',
    41: 'Net Profit Margin',
    42: 'Performance (Week)',
    43: 'Performance (Month)',
    44: 'Performance (Quarter)',
    45: 'Performance (Half Year)',
    46: 'Performance (Year)',
    47: 'Performance (YearToDate)',
    48: 'Beta',
    49: 'Average True Range',
    50: 'Volatility (Week)',
    51: 'Volatility (Month)',
    52: '20-Day Simple Moving Average',
    53: '50-Day Simple Moving Average',
    54: '200-Day Simple Moving Average',
    55: '50-Day High',
    56: '50-Day Low',
    57: '52-Week High',
    58: '52-Week Low',
    59: 'RSI',
    60: 'Change from Open',
    61: 'Gap',
    62: 'Analyst Recom.',
    63: 'Average Volume',
    64: 'Relative Volume',
    65: 'Price',
    66: 'Change',
    67: 'Volume',
    68: 'Earnings Date',
    69: 'Target Price',
    70: 'IPO Date'
}


class Custom(Overview):
    """Custom inherit from overview module.
    Getting information from the finviz screener custom page.
    """
    def __init__(self):
        """initiate module
        """
        self.BASE_URL = 'https://finviz.com/screener.ashx?v=151{signal}{filter}&ft=4{ticker}'
        self.url = self.BASE_URL.format(signal='', filter='', ticker='')
        Overview._loadSetting(self)

    def getColumns(self):
        """Get information about the columns
        Returns:
            columns(dict): return the index and column name.
        """
        return columns

    def _screener_helper(self, i, page, rows, df, num_col_index, table_header, limit):
        """Get screener table helper function.
        Returns:
            df(pandas.DataFrame): screener information table
        """
        if i == page - 1:
            df = self._get_table(rows, df, num_col_index, table_header, limit=((limit - 1) % 20 + 1))
        else:
            df = self._get_table(rows, df, num_col_index, table_header)
        return df

    def ScreenerView(self,
                     order='ticker',
                     limit=-1,
                     verbose=1,
                     ascend=True,
                     filters=["geo_usa", "sh_float_u100", "sh_price_u50", "ta_gap_u4"],
                     columns=[0, 1, 2, 3, 4, 5, 6, 7, 65, 66, 67]):
        """Get screener table.
        Args:
            order(str): sort the table by the choice of order.
            limit(int): set the top k rows of the screener.
            verbose(int): choice of visual the progress. 1 for visualize progress.
            ascend(bool): if True, the order is ascending.
            columns(list): columns of your choice. Default index: 0,1,2,3,4,5,6,7,65,66,67.
        Returns:
            df(pandas.DataFrame): screener information table
        """
        url = self.url
        if order != 'ticker':
            if order not in self.order_dict:
                raise ValueError()
            url = self.url+'&'+self.order_dict[order]
        if not ascend:
            url = url.replace('o=', 'o=-')
        columns = [str(i) for i in columns]
        url += '&f=' + ','.join(filters)
        url += '&c=' + ','.join(columns)
        soup = webScrap(url)

        page = self._get_page(soup)
        if page == 0:
            print('No ticker found.')
            return None

        if limit != -1:
            if page > (limit-1)//20+1:
                page = (limit-1)//20+1

        if verbose == 1:
            progressBar(1, page)
        table = soup.findAll('table')[18]
        rows = table.findAll('tr')
        table_header = [i.text for i in rows[0].findAll('td')][1:]
        num_col_index = [table_header.index(i) for i in table_header if i in NUMBER_COL]
        df = pd.DataFrame([], columns=table_header)
        df = self._screener_helper(0, page, rows, df, num_col_index, table_header, limit)

        for i in range(1, page):
            if verbose == 1:
                progressBar(i+1, page)

            url = self.url
            if order == 'ticker':
                url += '&r={}'.format(i * 20 + 1)
            else:
                url += '&r={}'.format(i * 20 + 1)+'&'+self.order_dict[order]
            if not ascend:
                url = url.replace('o=', 'o=-')
            url += '&c=' + ','.join(columns)
            soup = webScrap(url)
            table = soup.findAll('table')[18]
            rows = table.findAll('tr')
            df = self._screener_helper(i, page, rows, df, num_col_index, table_header, limit)
        return df
    
    
    
def build_day_trade_bet():
    tickers = []
    fiz = Custom()
    df_fiz = fiz.ScreenerView(order='ticker',
                              limit=-1,
                              verbose=1,
                              ascend=True,
                              filters=["geo_usa", "sh_float_u100", "sh_price_u50", "sh_curvol_o100", "ta_gap_u4"],
                              columns=[0, 1, 2, 3, 4, 5, 6, 7, 25, 61, 65, 66, 67])

    for index, row in df_fiz.iterrows():
    
        if row["Gap"] > 0.04:
            reward_minimun_amount = 0.02
            
            gap = str(round(row["Gap"] * 100, 2)) + " %"
    
            if row["Price"] > 10:
                reward_minimun_amount = 0.10
            elif row["Price"] > 5:
                reward_minimun_amount = 0.05
                
            industry = row["Industry"]
            sector = row ["Sector"]
                
            ticker = {
                			"ticker": row["Ticker"],
                			"stop_loss_percent": 1,
                			"reward_target_percent": 3,
                			"reward_minimun_amount": reward_minimun_amount,
                			"reward_profit_saved_percent": 50,
                			"stop_buying_after_hour": 15,
                			"gap": gap,
                			"volume_greater": 100000,
                         "sector": sector,
                         "industry": industry
               		}
            
            tickers.append(ticker)

    return tickers

def send_day_trade_setup(body):
    try:
        URL = "https://gcdekm9x7k.execute-api.us-west-1.amazonaws.com/default/setupInvestment"
        HEADERS = {"Content-Type": "application/json", "x-api-key": "ajWgX13WZ84rdArqDm7Jp1UjlUpSbOsX3Jxyv2Cx"}
        result = requests.post(url=URL, headers=HEADERS, json=body)
        statusOK = result.status_code == 200
    except:
        result = None
        statusOK = False
        
    return result.json(), statusOK


def build_day_trade_setup(idUser):
    
    betTickers = build_day_trade_bet()
       
    setup = {
        "user_id": idUser,
        "user_name": "Rodrigo Costa",
        "tickers": betTickers,
        "ticker": {},
        "earning_calendar": [],
        "stock_after_hour": False,
        "data_period": "2d",
        "data_interval": "1m",
        "date_start": "2021-03-05 00:00:00.345727",
        "date_end": "2021-03-06 00:00:00.345727",
        "day_trade_mode": True,
        "invest_amount": {
          "initial_amount": 1000,
          "total_amount": 1013,
          "ticker_amount": 500,
          "today_goal_amount": 50,
          "final_goal_amount": 2000
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
          "apply": True,
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
            "apply": False,
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
            }
      }
    
    result, statusOK = send_day_trade_setup(setup)
    
    return result, statusOK


result, statusOK = build_day_trade_setup("131efbb2-83a0-4786-9241-1625c8cb801a")
print("Status", statusOK)
print("Result", result)

#betTickers = build_day_trade_bet()