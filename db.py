
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("data", "stocksense.db")

def get_connection():
    """Return a connection to the SQLite database."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT,
            date TEXT,
            predicted_price REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS retrain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            retrain_time TEXT,
            model_version TEXT,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()

def save_prediction(company, date, predicted_price):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (company, date, predicted_price) VALUES (?, ?, ?)",
        (company, date, predicted_price)
    )
    conn.commit()
    conn.close()

def fetch_predictions(company):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT date, predicted_price FROM predictions WHERE company = ?", (company,))
    rows = cur.fetchall()
    conn.close()
    return rows