# finnhub_client.py
import finnhub
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

# Load .env API key
load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


def fetch_stock_data(symbol: str, resolution='D', count=30):
    """
    Fetch historical stock candle data from Finnhub.
    :param symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    :param resolution: Time resolution (1, 5, 15, 30, 60, D, W, M)
    :param count: Number of candles (e.g., last 30 days)
    :return: Pandas DataFrame with OHLC data
    """
    try:
        now = datetime.utcnow()
        if resolution == 'D':
            start = now - timedelta(days=count)
        elif resolution == 'W':
            start = now - timedelta(weeks=count)
        elif resolution == 'M':
            start = now - timedelta(days=30 * count)
        else:
            start = now - timedelta(days=count)

        from_timestamp = int(start.timestamp())
        to_timestamp = int(now.timestamp())

        res = finnhub_client.stock_candles(symbol, resolution, from_timestamp, to_timestamp)

        if res and res.get('s') == 'ok':
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(res['t'], unit='s'),
                'open': res['o'],
                'high': res['h'],
                'low': res['l'],
                'close': res['c'],
                'volume': res['v']
            })
            return df
        else:
            print(f"⚠️ No valid data returned for {symbol}")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()
