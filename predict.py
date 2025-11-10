# predict.py
import os
<<<<<<< HEAD
import joblib
=======
import pickle
>>>>>>> d03f45fdf4d60623d1875dfb441f39a4198e0c88
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model
from finnhub_client import fetch_stock_data

# ---------- Utility functions ----------
def load_scaler(symbol):
    scaler_path = f"models/{symbol}_scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found for {symbol}. Train first using train.py.")
    with open(scaler_path, "rb") as f:
<<<<<<< HEAD
        return joblib.load(scaler_path)
=======
        return pickle.load(f)
>>>>>>> d03f45fdf4d60623d1875dfb441f39a4198e0c88

def predict_future(symbol, time_steps=60):
    """Load model+scaler for `symbol`, fetch recent data, and predict next close."""
    model_path = f"models/{symbol}_lstm_model.h5"
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found for {symbol}. Skipping...")
        return None

    model = load_model(model_path)
    scaler = load_scaler(symbol)

    df = fetch_stock_data(symbol)
    if df.empty or 'close' not in df.columns:
        print(f"⚠️ No valid data for {symbol}. Skipping...")
        return None

    data = df['close'].values.reshape(-1, 1)
    if len(data) < time_steps:
        print(f"⚠️ Not enough data for {symbol}. Skipping...")
        return None

    scaled = scaler.transform(data)
    last_seq = scaled[-time_steps:]
    X_input = np.reshape(last_seq, (1, time_steps, 1))
    pred_scaled = model.predict(X_input)
    pred_price = scaler.inverse_transform(pred_scaled)
    predicted_value = float(pred_price[0][0])

    print(f"📈 Predicted next close for {symbol}: ${predicted_value:.2f}")
    return predicted_value

# ---------- Main multi-stock prediction ----------
if __name__ == "__main__":
    # Define all your tracked stocks here
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

    results = []
    for sym in symbols:
        pred = predict_future(sym)
        if pred is not None:
            results.append({"symbol": sym, "predicted_price": pred})

    if results:
        df_results = pd.DataFrame(results)
        output_path = "predictions.csv"
        df_results["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_results.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to {output_path}")
    else:
<<<<<<< HEAD
        print("\n⚠️ No predictions were generated. Please ensure models exist for each symbol.")
=======
        print("\n⚠️ No predictions were generated. Please ensure models exist for each symbol.")
>>>>>>> d03f45fdf4d60623d1875dfb441f39a4198e0c88
