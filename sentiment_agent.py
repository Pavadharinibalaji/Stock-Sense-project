import requests
from transformers import pipeline
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv 

load_dotenv()



sentiment_model = pipeline("sentiment-analysis")


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

def fetch_company_news(symbol):
    """Fetch recent company news from Finnhub for past 3 days."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3)
    params = {
        "symbol": symbol,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "token": FINNHUB_API_KEY
    }
    response = requests.get(FINNHUB_NEWS_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching news:", response.text)
        return []

def analyze_sentiment(news_list):
    """Analyze sentiment of news headlines using Hugging Face model."""
    sentiments = []
    for news in news_list[:10]:  
        result = sentiment_model(news.get("headline", ""))
        label = result[0]['label']
        score = result[0]['score']
        sentiments.append({"headline": news.get("headline"), "label": label, "score": score})
    return sentiments

def get_company_sentiment(symbol):
    """Main function to fetch and analyze sentiment."""
    news = fetch_company_news(symbol)
    if not news:
        return []
    sentiments = analyze_sentiment(news)
    return sentiments