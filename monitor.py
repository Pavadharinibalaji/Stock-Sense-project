# monitor.py
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime

def evaluate_model(symbol):
    pred = pd.read_csv(f"predictions/{symbol}.csv")
    actual = pd.read_csv(f"actuals/{symbol}.csv")

    rmse = mean_squared_error(actual['price'], pred['price'], squared=False)
    accuracy = accuracy_score(actual['direction'], pred['direction'])
    drift = accuracy < 0.80

    log = pd.DataFrame([{
        "timestamp": datetime.now(),
        "symbol": symbol,
        "rmse": rmse,
        "accuracy": accuracy,
        "drift": drift
    }])
    log.to_csv("logs/performance_log.csv", mode='a', header=not pd.io.common.file_exists("logs/performance_log.csv"), index=False)
    return drift