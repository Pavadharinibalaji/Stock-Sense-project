import streamlit as st
import pandas as pd
from finnhub_client import fetch_stock_data
from train import train_model
from predict import predict_future
from sentiment_agent import get_company_sentiment
from db import init_db, save_prediction, fetch_predictions
from retrain import retrain_all

# 🌐 App setup
st.set_page_config(page_title="📈 StockSense", layout="wide")
st.title("💹 StockSense – Cloud-Based AI Predictive Dashboard")

# Initialize database
init_db()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📊 Predict Stock", "📰 News Sentiment", "🔁 Retrain Models", "📜 Prediction History"]
)

# =============== PAGE 1 : STOCK PREDICTION ===============
if page == "📊 Predict Stock":
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY):", "AAPL").upper()

    if st.button("🔍 Predict Stock Price"):
        st.info(f"Fetching latest stock data for {symbol} ...")
        df = fetch_stock_data(symbol)

        if df is not None and not df.empty:
            try:
                # Train model (if not trained yet)
                model, scaler = train_model(symbol)

                # Predict future price
                predicted_price = predict_future(symbol, scaler)
                st.metric("Predicted Next Close Price", f"${predicted_price:.2f}")

                # Save prediction
                save_prediction(symbol, pd.Timestamp.now().strftime("%Y-%m-%d"), predicted_price)
                st.success("✅ Prediction saved to database")

            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
        else:
            st.warning("⚠️ No data available for this symbol. Please check the ticker.")

# =============== PAGE 2 : SENTIMENT ANALYSIS ===============
elif page == "📰 News Sentiment":
    symbol = st.text_input("Enter Stock Symbol for News:", "AAPL").upper()
    if st.button("🧠 Analyze Sentiment"):
        st.info(f"Fetching and analyzing latest news for {symbol} ...")
        sentiments = get_company_sentiment(symbol)
        if sentiments:
            df_sent = pd.DataFrame(sentiments)
            st.dataframe(df_sent)
        else:
            st.warning("⚠️ No news found or API limit reached.")

# =============== PAGE 3 : RETRAIN MODELS ===============
elif page == "🔁 Retrain Models":
    st.subheader("Weekly Model Retraining")
    st.write("Run model updates for all configured stocks (AAPL, MSFT, TSLA, etc.)")

    if st.button("🚀 Start Retraining"):
        with st.spinner("Retraining all stock models... please wait..."):
            retrain_all()
        st.success("✅ All models retrained successfully!")

# =============== PAGE 4 : VIEW HISTORY ===============
elif page == "📜 Prediction History":
    symbol = st.text_input("Enter Stock Symbol to View History:", "AAPL").upper()
    if st.button("📂 Load Prediction History"):
        data = fetch_predictions(symbol)
        if data:
            df_hist = pd.DataFrame(data, columns=["Date", "Predicted Price"])
            st.dataframe(df_hist)
        else:
            st.info("No previous predictions found for this stock.")
