import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from prepare_data import get_prepared_data

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(symbol):
    """Train a model for a specific company symbol"""
    print(f"🚀 Training LSTM model for {symbol}...")

    X, y, scaler, df = get_prepared_data(symbol)
    model = build_lstm_model((X.shape[1], 1))

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=25, batch_size=32, verbose=1, callbacks=[early_stop])

    # Create models directory if not exists
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{symbol}_lstm_model.h5")
    print(f"✅ Model for {symbol} saved as models/{symbol}_lstm_model.h5")

    return model, scaler
