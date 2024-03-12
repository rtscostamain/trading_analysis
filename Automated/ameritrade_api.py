# Import the client
from td.client import TDClient

# Create a new session, credentials path is required.
TDSession = TDClient(
    client_id='xxx',
    redirect_uri='https://localhost/rtscosta',
    credentials_path='D:\\Investimento\\trading_analysis\\Automated\\ameri_cre.txt',
)

# Login to the session
TDSession.login()

# Grab real-time quotes for 'MSFT' (Microsoft)
msft_quotes = TDSession.get_quotes(instruments=['MSFT'])

# Grab real-time quotes for 'AMZN' (Amazon) and 'SQ' (Square)
multiple_quotes = TDSession.get_quotes(instruments=['AMZN','SQ'])


