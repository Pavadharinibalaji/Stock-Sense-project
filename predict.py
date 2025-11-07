import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from finnhub_client import fetch_stock_data

def predict_future(symbol, scaler, time_steps=60):
    """Predict the next closing price for the given stock"""
    model_path = f"models/{symbol}_lstm_model.h5"
    try:
        model = load_model(model_path)
    except:
        raise FileNotFoundError(f"Model not found for {symbol}. Please train it first using train.py.")

    df = fetch_stock_data(symbol)
    data = df['close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    last_sequence = scaled_data[-time_steps:]
    X_input = np.reshape(last_sequence, (1, time_steps, 1))
    pred_scaled = model.predict(X_input)
    pred_price = scaler.inverse_transform(pred_scaled)

    print(f"📈 Predicted next close for {symbol}: ${pred_price[0][0]:.2f}")
    return float(pred_price[0][0])
