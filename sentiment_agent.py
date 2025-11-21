# sentiment_agent.py
import requests
from transformers import pipeline
import os
from datetime import date, timedelta
from dotenv import load_dotenv

# -------------------- SETUP --------------------

# Load environment variables from .env file
load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/news"

# Best financial sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)

# Normalize sentiments
def normalize_label(label):
    label = label.lower()
    if label == "positive":
        return "POSITIVE", 1
    if label == "negative":
        return "NEGATIVE", -1
    return "NEUTRAL", 0


# -------------------- FETCH NEWS --------------------
def fetch_general_news(symbol=None):
    today = date.today()
    from_date = (today - timedelta(days=7)).isoformat()
    to_date = today.isoformat()

    params = {"category": "general", "token": FINNHUB_API_KEY}

    print(f"Fetching general news from {from_date} to {to_date}...")

    response = requests.get(FINNHUB_NEWS_URL, params=params)
    if response.status_code != 200:
        print("❌ Error fetching news:", response.text)
        return []

    data = response.json()

    if symbol:
        symbol = symbol.lower()
        data = [
            n for n in data
            if symbol in n.get("headline", "").lower()
            or symbol in n.get("summary", "").lower()
        ]

    return data[:15]


# -------------------- ANALYZE SENTIMENT --------------------
def analyze_sentiment(news_list):
    sentiments = []
    for news in news_list[:10]:
        headline = news.get("headline", "").strip()
        if not headline:
            continue

        try:
            result = sentiment_model(headline)
            if result and isinstance(result, list) and "label" in result[0]:
                raw_label = result[0]["label"]
                label, score = normalize_label(raw_label)
                confidence = result[0]["score"]
            else:
                label, score, confidence = "NEUTRAL", 0, 0.0

        except Exception as e:
            print(f"⚠️ Sentiment failed: {headline} | Error: {e}")
            label, score, confidence = "ERROR", 0, 0.0

        sentiments.append({
            "headline": headline,
            "label": label,
            "score": score,
            "confidence": confidence
        })

    return sentiments


# -------------------- MAIN FUNCTION --------------------
def get_general_sentiment(symbol):
    news = fetch_general_news(symbol)
    if not news:
        return [{
            "headline": "No relevant news found",
            "label": "-",
            "score": 0,
            "confidence": 0.0
        }]

    return analyze_sentiment(news)
