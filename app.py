import streamlit as st
import plotly.graph_objects as go
from sentiment_agent import get_company_sentiment
from finnhub_client import fetch_stock_data
from keras.models import load_model
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Stock Sense Dashboard",
    page_icon="assets/Stock-sense logo(1).jpeg",
    layout="wide"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        body {
            background-color: #1e0033;
            color: #00ffff;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #2b0052, #1e0033);
            color: #00ffff;
        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        [data-testid="stSidebar"] {
            background-color: #2b0052;
        }
        h1, h2, h3, h4 {
            color: #00ffff !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #7a00ff;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 25px;
            border: 2px solid #00ffff;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #00ffff;
            color: #1e0033;
            border-color: #7a00ff;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)



# ---------------------- HELPER FUNCTIONS ----------------------
def load_scaler(symbol):
    try:
        return joblib.load(f"models/{symbol}_scaler.pkl")
    except:
        return None


@st.cache_resource
def load_lstm_model(symbol):
    try:
        return load_model(f"models/{symbol}_lstm_model.h5")
    except:
        return None


def predict_future(symbol, time_steps=60):
    df = fetch_stock_data(symbol)

    if df is None or df.empty or "close" not in df.columns:
        return None

    model = load_lstm_model(symbol)
    scaler = load_scaler(symbol)

    if model is None or scaler is None:
        return None

    data = df["close"].values.reshape(-1, 1)
    if len(data) < time_steps:
        return None

    scaled = scaler.transform(data)
    seq = scaled[-time_steps:]
    X_input = np.reshape(seq, (1, time_steps, 1))

    pred_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

    return predicted_price, df



# ---------------------- HEADER ----------------------
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image("assets/Stock-sense logo(1).jpeg", width=100)
with col2:
    st.title("Stock Sense – Smart AI Stock Predictor")

st.markdown("---")


# ---------------------- SYMBOL SELECTOR ----------------------
st.subheader("Search and Select a Stock Symbol")

symbol = st.selectbox(
    "Choose a stock symbol to analyze:",
    ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META", "NFLX", "INFY", "TCS"],
    index=0
)

st.markdown("---")


# ---------------------- ACTION BUTTONS ----------------------
col_a, col_b = st.columns([0.5, 0.5])
with col_a:
    predict_button = st.button("Predict Stock Price")
with col_b:
    sentiment_button = st.button("View Sentiment Analysis")

st.markdown("---")



# ---------------------- PREDICTION SECTION ----------------------
if predict_button:
    st.subheader(f"Stock Prediction for {symbol}")

    result = predict_future(symbol)

    if result is None:
        st.error("Model missing or data unavailable. Train the model first!")
    else:
        predicted_price, df = result

        last_price = df["close"].iloc[-1]
        trend = "🔺 UP" if predicted_price > last_price else "🔻 DOWN"
        color = "#00ff00" if predicted_price > last_price else "#ff0055"

        # ---- Display Metrics ----
        st.markdown(f"""
            <h3 style='color:{color}'>
                Predicted Next Close: ${predicted_price:.2f} {trend}
            </h3>
        """, unsafe_allow_html=True)

        st.metric(
            label="Current Price",
            value=f"${last_price:.2f}",
            delta=f"{(predicted_price - last_price):.2f}"
        )

        # ---- Plot Chart ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["close"],
            mode='lines',
            name='Actual Prices',
            line=dict(color='cyan', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[df.index[-1], df.index[-1] + pd.Timedelta(days=1)],
            y=[last_price, predicted_price],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='magenta', width=3, dash='dot')
        ))

        fig.update_layout(
            title=f"{symbol} Stock Price Prediction",
            template="plotly_dark",
            paper_bgcolor='#1e0033',
            plot_bgcolor='#1e0033',
            font=dict(color='#00ffff')
        )

        st.plotly_chart(fig, use_container_width=True)



# ---------------------- SENTIMENT SECTION ----------------------
if sentiment_button:
    st.subheader(f"Sentiment Overview for {symbol}")

    try:
        sentiment_data = get_company_sentiment(symbol)
        df = pd.DataFrame(sentiment_data)
        
        st.dataframe(df.style.set_properties(**{
            'background-color': '#2b0052',
            'color': '#00ffff',
            'border-color': '#00ffff'
        }))
        
    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")


# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>© 2025 Stock Sense | Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)