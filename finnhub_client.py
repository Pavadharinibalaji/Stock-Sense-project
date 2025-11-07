# finnhub_client.py
import os
import finnhub
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

if not API_KEY:
    raise ValueError("❌ FINNHUB_API_KEY not found in .env file. Please add it to your .env file.")

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=API_KEY)

def fetch_stock_data(symbol):
    """
    Fetch latest stock candle data for the given stock symbol.
    Returns a DataFrame with 'date', 'open', 'high', 'low', 'close', 'volume'.
    """
    try:
        res = finnhub_client.quote(symbol)
        if not res or 'c' not in res:
            print(f"❌ Error fetching stock data for {symbol}: Invalid response.")
            return pd.DataFrame()

        data = {
            'symbol': [symbol],
            'current': [res['c']],
            'high': [res['h']],
            'low': [res['l']],
            'open': [res['o']],
            'previous_close': [res['pc']]
        }

        df = pd.DataFrame(data)
        print(f"✅ Latest data fetched successfully for {symbol}")
        return df

    except Exception as e:
        print(f"❌ Error fetching stock data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_general_news(category='general', count=20):
    """
    Fetch general market news (not company-specific).
    Default category: 'general', count: 20
    """
    try:
        news = finnhub_client.general_news(category, min_id=0)
        if not news:
            print("⚠️ No news found.")
            return pd.DataFrame()

        df = pd.DataFrame(news[:count])
        df = df[['headline', 'summary', 'source', 'datetime', 'url']]
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

        print(f"✅ Fetched {len(df)} general news articles.")
        return df

    except Exception as e:
        print(f"❌ Error fetching general news: {e}")
        return pd.DataFrame()


# For testing
if __name__ == "__main__":
    print("\n🔹 Testing stock data fetch:")
    print(fetch_stock_data("AAPL"))

    print("\n🔹 Testing general news fetch:")
    print(fetch_general_news())