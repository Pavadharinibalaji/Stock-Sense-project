import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from finnhub_client import fetch_stock_data  # Make sure this function accepts a 'symbol' argument

def prepare_lstm_data(df, time_steps=60):
    """Convert raw stock data into time-series sequences for LSTM"""
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def get_prepared_data(symbol, time_steps=60):
    """Fetch and prepare data for a given stock symbol"""
    df = fetch_stock_data(symbol)
    X, y, scaler = prepare_lstm_data(df, time_steps)
    return X, y, scaler, df
