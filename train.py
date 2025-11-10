# train.py
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from finnhub_client import fetch_stock_data
from prepare_data import prepare_lstm_data
import joblib  # For saving the scaler

# Directory to save models
os.makedirs("models", exist_ok=True)

# List of stock symbols to train
STOCK_LIST = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NFLX", "INFY"]

def build_lstm_model(input_shape):
    """Define and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model_for_symbol(symbol):
    """Train and save model for one company."""
    print(f"\n🚀 Training LSTM model for {symbol}...")

    df = fetch_stock_data(symbol)
    if df.empty:
        print(f"⚠️ No data available for {symbol}, skipping.")
        return

    X_train, y_train, scaler = prepare_lstm_data(df)
    model = build_lstm_model((X_train.shape[1], 1))

    # Early stopping to prevent overfitting
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[es])

    model_path = f"models/{symbol}_lstm_model.h5"
    scaler_path = f"models/{symbol}_scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Model and scaler saved for {symbol}:")
    print(f"   - Model: {model_path}")
    print(f"   - Scaler: {scaler_path}")

def train_all_stocks():
    """Train all stocks in the list sequentially."""
    for symbol in STOCK_LIST:
        try:
            train_model_for_symbol(symbol)
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}")

if __name__ == "__main__":
    train_all_stocks()
