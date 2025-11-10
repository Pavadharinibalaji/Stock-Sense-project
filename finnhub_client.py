# finnhub_client.py

import os
import finnhub
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 🔹 Load API key
import os
import finnhub
import pandas as pd
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

if not API_KEY:

    print("⚠️  FINNHUB_API_KEY not found in .env — Finnhub will be skipped for data fetching.")

# 🔹 Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=API_KEY) if API_KEY else None


# ===========================================
#  FETCH FROM FINNHUB
# ===========================================
def fetch_from_finnhub(symbol, days_back=365):
    """Try fetching historical data from Finnhub (if API allows)."""
    try:
        if not finnhub_client:
            return None

        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())

        res = finnhub_client.stock_candles(symbol, 'D', start_time, end_time)
        if res.get("s") != "ok":
            print(f"⚠️ Finnhub returned: {res.get('s')} for {symbol}")
            return None

        df = pd.DataFrame(res)
        df.rename(columns={'c': 'close', 'o': 'open', 'h': 'high', 'l': 'low', 'v': 'volume', 't': 'timestamp'}, inplace=True)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('date')

        if len(df) < 50:  # Too few records
            return None

        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"❌ Finnhub error for {symbol}: {e}")
        return None


# ===========================================
#  FETCH FROM YFINANCE
# ===========================================
def fetch_from_yfinance(symbol, period="1y"):
    """Fallback: Fetch data from Yahoo Finance (always available)."""
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        if df.empty:
            print(f"⚠️ No data found for {symbol} using yfinance.")
            return None

        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                           'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df = df.sort_values('date')
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"❌ YFinance error for {symbol}: {e}")
        return None


# ===========================================
#  HYBRID FETCH FUNCTION
# ===========================================
def fetch_stock_data(symbol, days_back=365):
    """
    Fetch stock data — try Finnhub first, then fallback to Yahoo Finance.
    Ensures consistent DataFrame for LSTM model.
    """
    print(f"\n📊 Fetching stock data for {symbol}...")

    # 1️⃣ Try Finnhub
    df = fetch_from_finnhub(symbol, days_back)
    if df is not None and not df.empty:
        print(f"✅ Using Finnhub data for {symbol} ({len(df)} records)")
        return df

    # 2️⃣ Fallback to yfinance
    df = fetch_from_yfinance(symbol, period="2y")
    if df is not None and not df.empty:
        print(f"✅ Using Yahoo Finance data for {symbol} ({len(df)} records)")
        return df

    # 3️⃣ No data at all
    print(f"❌ Could not fetch data for {symbol} from either source.")
    return pd.DataFrame()


# ===========================================
#  TEST
# ===========================================
if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG"]
    for sym in symbols:
        df = fetch_stock_data(sym)
        print(df.head())

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
