import pandas as pd
from datetime import datetime
from datetime import timedelta
from yahoo_earnings_calendar import YahooEarningsCalendar
import dateutil.parser
import requests
from finvizfinance.util import webScrap, numberCovert
from finvizfinance.quote import finvizfinance


def get_earning_calendar(days_prior, days_after):
    # setting the dates
    start_date = (datetime.now().date() - timedelta(days=days_prior))
    end_date = (datetime.now().date() + timedelta(days=days_after))
    
    # downloading the earnings calendar
    yec = YahooEarningsCalendar()
    earnings_list = yec.earnings_between(start_date, end_date)
    return earnings_list

def get_sector_info():
    url = 'https://finviz.com/groups.ashx?{group}&v=110'.format(group='g=sector')
    soup = webScrap(url)
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
    group_dict = dict(zip(key, value))     
    
    result = {}
    
    return result

def get_investor_info(userId):
    try:
        URL = "https://gcdekm9x7k.execute-api.us-west-1.amazonaws.com/default/investimentInfo?user_id={0}".format(userId)
        HEADERS = {"x-api-key": "ajWgX13WZ84rdArqDm7Jp1UjlUpSbOsX3Jxyv2Cx"}
        result = requests.get(url=URL, headers=HEADERS)
        statusOK = result.status_code == 200
    except:
        result = None
        statusOK = False
    return result.json(), statusOK

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

def get_ticker_fundamental(ticker):
    stock = finvizfinance(ticker)
    stock_fundament = stock.TickerFundament()
    return stock_fundament


def build_tickers_earning(df):
    tickers = []
    
    DF = df.copy()
    
    for index, row in DF.iterrows():
    
        startdatetimetype = ""    
    
        if row["startdatetimetype"] == "BMO":
            startdatetimetype = "Before Market Open"
        elif row["startdatetimetype"] == "AMC":
            startdatetimetype = "After Market Close"
        else:
            startdatetimetype = row["startdatetimetype"]
    
        try:
            ticker_fundamental = get_ticker_fundamental(row["ticker"])
        except:
            ticker_fundamental["Sector"] = ""
            ticker_fundamental["Industry"] = ""
            
        
        #stock = finvizfinance(row["ticker"])
        #ticker_fundamental = stock.TickerFundament()
        
        print(row["ticker"], ticker_fundamental["Sector"])
    
        ticker = {
            			"ticker": row["ticker"],
            			"company": row["companyshortname"],
                   "sector": ticker_fundamental["Sector"],
                   "industry": ticker_fundamental["Industry"],
                   "event_date": row["startdatetime"],
                   "call_time": startdatetimetype,
                   "eps_estimated": None if pd.isnull(row["epsestimate"]) else row["epsestimate"],
                   "eps_reported": None if pd.isnull(row["epsactual"]) else row["epsactual"]
           		}
        
        tickers.append(ticker)
    
    print(tickers)
    
    return tickers

# saving the data in a pandas DataFrame
weekno = datetime.today().weekday()
start_day = 0
end_day = 2 if weekno == 5 else 1


earnings_list = get_earning_calendar(start_day,end_day)
earnings_df = pd.DataFrame(earnings_list)
earnings_df.head()

temp = build_tickers_earning(earnings_df)

stock = finvizfinance('DDD')
stock_fundament = stock.TickerFundament()
print(stock_fundament["Sector"])
print(stock_fundament)


"""
userId = "131efbb2-83a0-4786-9241-1625c8cb801a"
print("Loading API data for user {0}".format(userId))
params, status = get_investor_info(userId)
if status:
    print("API data loaded")
else:
    print("Error loading API")

# saving the data in a pandas DataFrame
weekno = datetime.today().weekday()
start_day = 0
end_day = 2 if weekno == 5 else 1


earnings_list = get_earning_calendar(start_day,end_day)
earnings_df = pd.DataFrame(earnings_list)
earnings_df.head()

params["earning_calendar"] = build_tickers_earning(earnings_df)

#result, statusOK = send_day_trade_setup(params)
#print("API Updated")
"""


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
        
def create_sector_performance():
    result = []
    top = []
    df_sectors = Performance().ScreenerView()    
    df_sectors = df_sectors.sort_values(by=['Perf Week'], ascending=False)
    loop = 0

    for index, row in df_sectors.iterrows():
        
        if loop < 3:
           top.append(row["Name"]) 
        
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
        loop += 1
    
        
    return result, top

teste, top = create_sector_performance()
print(teste)
print(top)

stock = finvizfinance('DDD')
stock_fundament = stock.TickerFundament()
print(stock_fundament)
