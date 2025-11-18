import os
import pandas as pd
from datetime import datetime
from keras.models import load_model
from finnhub_client import fetch_stock_data
from prepare_data import prepare_lstm_data
from train import train_model_for_symbol
from db import get_connection

# 🔁 List of stocks to retrain weekly (expand as you wish)
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX", "INFY", "TCS"]

def retrain_model(symbol: str):
    """Retrain or fine-tune the model using latest data for a single stock."""
    print(f"\n🔄 Starting retraining for {symbol}...")

    # 1️⃣ Fetch fresh stock data
    new_data = fetch_stock_data(symbol)
    if new_data is None or new_data.empty:
        print(f"⚠️ No data fetched for {symbol}. Skipping retrain.")
        return

    # 2️⃣ Prepare LSTM data
    try:
        X_train, y_train, scaler = prepare_lstm_data(new_data)
    except Exception as e:
        print(f"❌ Error preparing data for {symbol}: {e}")
        return

    # 3️⃣ Ensure model directory exists
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{symbol}_lstm_model.h5")

    # 4️⃣ Load model or train a new one
    if os.path.exists(model_path):
        print(f"📦 Found existing model for {symbol}. Fine-tuning...")
        model = load_model(model_path)
    else:
        print(f"🆕 No model found for {symbol}. Training from scratch...")
        model = train_model_for_symbol(symbol)

    # 5️⃣ Continue training with new data (fine-tuning)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    model.save(model_path)
    print(f"✅ Retraining completed for {symbol} → saved at {model_path}")

    # 6️⃣ Log retraining info into the database
    try:
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO retrain_logs (retrain_time, model_version, notes)
            VALUES (?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                f"{symbol}_v{datetime.now().strftime('%Y%m%d')}",
                f"Weekly retrain completed for {symbol}",
            ),
        )
        conn.commit()
        conn.close()
        print(f"🗂️ Logged retrain for {symbol} in database.")
    except Exception as e:
        print(f"⚠️ Failed to log retrain in database for {symbol}: {e}")


def retrain_all():
    """Retrain models for all configured stock symbols."""
    print("\n🚀 Starting retraining for all configured stocks...\n")
    for symbol in STOCK_LIST:
        try:
            retrain_model(symbol)
        except Exception as e:
            print(f"❌ Error retraining {symbol}: {e}")
    print("\n✅ All retraining tasks completed!\n")


if __name__ == "__main__":
    retrain_all()
