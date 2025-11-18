import streamlit as st
import plotly.graph_objects as go
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
from sentiment_agent import get_general_sentiment
from query_agent import run_query_agent








# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Stock Sense Dashboard",
    page_icon="assets/Stock-sense logo(1).jpeg",
    layout="wide"
)

# ---------------------- CUSTOM CSS (Premium UI) ----------------------
st.markdown("""
    <style>

    /* Main Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a0030 0%, #120022 40%, #0d001a 100%);
        color: #00ffff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #2b0052;
        border-right: 2px solid #7a00ff;
    }

    h1, h2, h3, h4, h5 {
        color: #00ffff !important;
        font-family: "Segoe UI", sans-serif;
    }

    /* Card Containers */
    .custom-box {
        padding: 25px;
        border-radius: 15px;
        background-color: rgba(43, 0, 82, 0.55);
        border: 1px solid #7a00ff;
        box-shadow: 0 0 20px rgba(122, 0, 255, 0.4);
        margin-bottom: 25px;
    }

    /* TABS */
    .stTabs [role="tab"] {
        background: #2b0052;
        color: #00ffff !important;
        font-size: 16px;
        border-radius: 10px;
        margin-right: 8px;
        padding: 10px 20px;
        border: 1px solid #00ffff;
    }
    .stTabs [aria-selected="true"] {
        background: #00ffff !important;
        color: #1e0033 !important;
        font-weight: 800 !important;
    }

    /* Toggle Switch */
    .switch {
      position: relative;
      display: inline-block;
      width: 55px;
      height: 28px;
    }
    .switch input {display:none;}
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #7a00ff;
      transition: .4s;
      border-radius: 34px;
    }
    .slider:before {
      position: absolute;
      content: "";
      height: 22px; width: 22px;
      left: 3px; bottom: 3px;
      background-color: white;
      transition: .4s;
      border-radius: 50%;
    }
    input:checked + .slider {
      background-color: #00ffff;
    }
    input:checked + .slider:before {
      transform: translateX(26px);
    }

    /* Sentiment Badges */
    .badge {
        padding: 6px 12px;
        font-size: 14px;
        border-radius: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .positive { background: #00ff9d; color:#00331e; }
    .negative { background: #ff0066; color:#330010; }
    .neutral  { background: #ffaa00; color:#332200; }

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

# ---------------------- UPGRADED HEADER ----------------------
img_base64 = "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"

header_html = f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 10px 20px;
        background-color: black;
        border-radius: 10px;
        box-shadow: 0 0 25px cyan;    /* Neon Glow */
    ">
        <img src="data:image/jpeg;base64,{img_base64}" 
             style="width:80px; border-radius:10px; box-shadow:0 0 20px cyan;">
        <h1 style="color: cyan; margin: 0; text-shadow: 0 0 10px cyan;">
            Stock Sense – Smart AI Stock Predictor
        </h1>
    </div>
"""

st.markdown(header_html, unsafe_allow_html=True)
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
    with st.container():   # <<< CONTAINER START
        st.subheader(f"📈 Stock Prediction for {symbol}")

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
                <h3 style='color:{color}; margin-top: 10px;'>
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
    # <<< CONTAINER END


# ---------------------- SENTIMENT TAB ----------------------
with tab2:
    with st.container():   # <<< SENTIMENT CONTAINER START
        st.subheader(f"📰 Sentiment Overview for General Market")

        try:
            sentiment_data = get_general_sentiment(symbol)
            df_sent = pd.DataFrame(sentiment_data)

            # --- Filter headlines containing the stock symbol ---
            filtered_data = [
                item for item in sentiment_data
                if symbol.lower() in item["headline"].lower()
            ]

            if filtered_data:
                # 🧠 AI agent
                response = run_query_agent(symbol, filtered_data)

                st.markdown("### 🧠 Agent Summary")
                st.write(response)

                st.markdown("### 📰 Matching Headlines")
                for item in filtered_data:
                    st.write(f"📰 {item['headline']}")
                    st.write(f"Sentiment: {item['label']} (Score: {item['score']:.2f})")
                    st.markdown("---")
            else:
                st.info(f"No headlines mentioning {symbol} found.")

            # ---- Display Sentiment Table ----
            st.markdown("### 📊 Full Sentiment Table")
            st.dataframe(df_sent.style.set_properties(**{
                'background-color': '#2b0052',
                'color': '#00ffff',
                'border-color': '#00ffff'
            }))

            # ---- Sentiment Chart ----
            if "label" in df_sent.columns:
                sentiment_counts = df_sent["label"].value_counts()
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

        st.title("AI Stock Sentiment Dashboard")

        if st.button("Analyze General News"):
            with st.spinner("Analyzing sentiment..."):
                data = get_general_sentiment(symbol)
            st.success("Analysis Complete!")
            for item in data:
                st.write(f"📰 {item['headline']}")
                label = item.get('label') or item.get('sentiment') or item.get('labels') or 'Unknown'
                score = item.get('score', 0)
                st.write(f"Sentiment: {label} (Score: {score:.2f})")
                st.divider()
    # <<< SENTIMENT CONTAINER END


# ---------------------- MODEL PERFORMANCE TAB ----------------------
with tab3:
    with st.container():   # <<< PERFORMANCE CONTAINER START
        st.subheader(f"📊 Model Performance for {symbol}")

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

            # Generate chart after retraining
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
    # <<< PERFORMANCE CONTAINER END


# ---------------------- TECHNICAL INDICATORS TAB ----------------------
with tab4:
    with st.container():   # <<< TECHNICAL CONTAINER START
        st.subheader(f"📉 Technical Indicators for {symbol}")

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
    # <<< TECHNICAL CONTAINER END



# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>© 2025 Stock Sense | Built by ❤️ PAVADHARINI B G, SANJANA V</p>",
    unsafe_allow_html=True
)
