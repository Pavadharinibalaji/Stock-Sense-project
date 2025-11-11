import streamlit as st
import plotly.graph_objects as go
from sentiment_agent import get_company_sentiment
from finnhub_client import fetch_stock_data
from keras.models import load_model
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from indicators import add_bollinger_bands, add_rsi, add_macd
import plotly.graph_objects as go
import os


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
    

def load_model_metrics(symbol):
    """Load saved model performance metrics from JSON file."""
    try:
        with open(f"models/{symbol}_metrics.json", "r") as f:
            return json.load(f)
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
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NFLX", "INFY"],
    index=0
)

st.markdown("---")

# ---------------------- TABS ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Prediction", "📰 Sentiment Analysis", "⚙️ Model Performance", "📊 Technical Indicators"

])

# ---------------------- PREDICTION TAB ----------------------
with tab1:
    st.subheader(f"Stock Prediction for {symbol}")

    result = predict_future(symbol)

    if result is None:
        st.error("Model missing or data unavailable. Train the model first!")
    else:
        predicted_price, df = result
        predicted_price = float(predicted_price)    
        last_price = float(df["close"].iloc[-1])
        trend = "🔺 UP" if predicted_price > last_price else "🔻 DOWN"
        color = "#00ff00" if predicted_price > last_price else "#ff0055"

        # ---- Display Metrics ----
        st.metric(
            label="Current Price",
            value=f"${last_price:.2f}",
            delta=f"{(predicted_price - last_price):.2f}"
        )

        st.markdown(f"""
            <h3 style='color:{color}'>
                Predicted Next Close: ${predicted_price:.2f} {trend}
            </h3>
        """, unsafe_allow_html=True)

        # ---- Candlestick Chart with Moving Averages ----
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick'
        )])

        # Add Moving Averages
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()

        fig_candle.add_trace(go.Scatter(
            x=df.index,
            y=df['MA10'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='MA10'
        ))

        fig_candle.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='MA50'
        ))
        # Prediction marker
        last_date = pd.to_datetime(df["date"].iloc[-1])
        next_date = last_date + pd.Timedelta(days=1)

        # Add prediction marker
        fig_candle.add_trace(go.Scatter(
            x=[last_date ,next_date],
            y=[predicted_price],
            mode='markers+text',
            text=[f"Predicted: ${predicted_price:.2f}"],
            textposition="top center",
            marker=dict(color='magenta', size=12),
            name='Prediction'
        ))

        fig_candle.update_layout(
            title=f"{symbol} Candlestick Chart with MA10 & MA50 + Prediction",
            template="plotly_dark",
            paper_bgcolor='#1e0033',
            plot_bgcolor='#1e0033',
            font=dict(color='#00ffff'),
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig_candle, use_container_width=True)

# ---------------------- SENTIMENT TAB ----------------------
with tab2:
    st.subheader(f"Sentiment Overview for {symbol}")
    try:
        sentiment_data = get_company_sentiment(symbol)
        df_sent = pd.DataFrame(sentiment_data)

        st.dataframe(df_sent.style.set_properties(**{
            'background-color': '#2b0052',
            'color': '#00ffff',
            'border-color': '#00ffff'
        }))

        # ---- Sentiment Chart ----
        if "sentiment" in df_sent.columns:
            sentiment_counts = df_sent["sentiment"].value_counts()
            fig_sent = go.Figure(data=[go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=['#00ff00', '#ff0055', '#ffaa00']
            )])
            fig_sent.update_layout(
                title="Sentiment Distribution",
                template="plotly_dark",
                paper_bgcolor='#1e0033',
                plot_bgcolor='#1e0033',
                font=dict(color='#00ffff')
            )
            st.plotly_chart(fig_sent, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching sentiment: {e}")

# ---------------------- MODEL PERFORMANCE TAB ----------------------
with tab3:
    st.subheader(f"Model Performance for {symbol}")

    metrics = load_model_metrics(symbol)

    if metrics is None:
        st.warning("No performance data found. Train the model to generate metrics.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['rmse']:.4f}")
        col2.metric("MAE", f"{metrics['mae']:.4f}")
        col3.metric("MAPE", f"{metrics['mape']:.2f}%")

        st.markdown("---")
        st.markdown(f"""
        **Training Info**
        - Trained on: `{metrics['trained_on']}`
        - Data points used: `{metrics['data_points']}`
        - Model file: `models/{symbol}_lstm_model.h5`
        """)


    # Calculate model age
    last_trained = datetime.strptime(metrics['trained_on'], "%Y-%m-%d %H:%M:%S")
    days_since = (datetime.now() - last_trained).days

    # Show freshness status
    if days_since > 30:
        st.warning(f"⚠️ Model is {days_since} days old. Consider retraining.")
    elif days_since > 7:
        st.info(f"ℹ️ Model is {days_since} days old. Still reasonably fresh.")
    else:
        st.success(f"✅ Model is fresh (trained {days_since} days ago).")

    st.markdown("---")

    st.download_button(
    label="📥 Download Metrics",
    data=json.dumps(metrics, indent=4),
    file_name=f"{symbol}_metrics.json",
    mime="application/json"
)
    
    if st.button("🔁 Retrain Model"):
        with st.spinner("Retraining model..."):
            os.system(f"python train.py --symbol {symbol}")
        st.success("Model retrained successfully!")
        metrics = load_model_metrics(symbol)



        # ✅ Load model, scaler, and data
        try:
            model = load_lstm_model(symbol)
            scaler = load_scaler(symbol)
            df = fetch_stock_data(symbol)

            if model and scaler and df is not None and not df.empty:
                from prepare_data import prepare_lstm_data
                X_train, y_train, scaler, X_test, y_test = prepare_lstm_data(df, return_test=True)
                y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
                y_pred = scaler.inverse_transform(model.predict(X_test))

                x = np.arange(len(y_true))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y_true.flatten(), name='Actual Prices', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=x, y=y_pred.flatten(), name='Predicted Prices', line=dict(color='orange')))
                fig.update_layout(
                    title='Predicted vs Actual Closing Prices',
                    xaxis_title='Time Steps',
                    yaxis_title='Price',
                    template='plotly_dark',
                    paper_bgcolor='#1e0033',
                    plot_bgcolor='#1e0033',
                    font=dict(color='#00ffff'),
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate prediction chart: {e}")

   

# ---------------------- TECHNICAL INDICATORS TAB ----------------------
with tab4:
    st.subheader(f"Technical Indicators for {symbol}")

    df = fetch_stock_data(symbol)
    if df is None or df.empty or "close" not in df.columns:
        st.error("Unable to fetch data for indicators.")
    else:
        df = add_bollinger_bands(df)
        df = add_rsi(df)
        df = add_macd(df)

        # ---- Bollinger Bands Chart ----
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Close', line=dict(color='cyan')))
        fig_bb.add_trace(go.Scatter(x=df['date'], y=df['Upper'], name='Upper Band', line=dict(color='orange')))
        fig_bb.add_trace(go.Scatter(x=df['date'], y=df['Lower'], name='Lower Band', line=dict(color='orange')))
        fig_bb.update_layout(title="Bollinger Bands", template="plotly_dark", paper_bgcolor='#1e0033', plot_bgcolor='#1e0033', font=dict(color='#00ffff'))
        st.plotly_chart(fig_bb, use_container_width=True)

        # ---- RSI Chart ----
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='lime')))
        fig_rsi.add_hline(y=70, line=dict(color='red', dash='dot'))
        fig_rsi.add_hline(y=30, line=dict(color='green', dash='dot'))
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", template="plotly_dark", paper_bgcolor='#1e0033', plot_bgcolor='#1e0033', font=dict(color='#00ffff'))
        st.plotly_chart(fig_rsi, use_container_width=True)

        # ---- MACD Chart ----
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='magenta')))
        fig_macd.add_trace(go.Scatter(x=df['date'], y=df['Signal'], name='Signal Line', line=dict(color='yellow')))
        fig_macd.update_layout(title="MACD & Signal Line", template="plotly_dark", paper_bgcolor='#1e0033', plot_bgcolor='#1e0033', font=dict(color='#00ffff'))
        st.plotly_chart(fig_macd, use_container_width=True)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>© 2025 Stock Sense | Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)