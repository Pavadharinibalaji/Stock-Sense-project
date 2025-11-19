import streamlit as st
import pyrebase
import time
from st_pages import hide_pages
import os

# -----------------------------------------------------------
# LOAD API KEYS FROM RENDER ENV
# -----------------------------------------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    print("⚠️ FINNHUB_API_KEY is missing in Render Environment Variables!")

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Stock Sense - Login",
    layout="centered",
    initial_sidebar_state="collapsed"
)

hide_pages(["dashboard"])

# -----------------------------------------------------------
# UI STYLES
# -----------------------------------------------------------
st.markdown("""
<style>
/* (Your CSS unchanged) */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a0030 0%, #120022 40%, #0d001a 100%);
    color: #00ffff !important;
}
.title {
    color: #00ffff !important;
    font-size: 2.7rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: -5px;
    text-shadow: 0px 0px 25px #00ffff;
    font-family: "Segoe UI", sans-serif;
}
.subtitle {
    color: #7aeaff !important;
    text-align: center;
    margin-bottom: 30px;
    font-size: 1.1rem;
    font-family: "Segoe UI", sans-serif;
}
.box {
    padding: 35px;
    border-radius: 18px;
    width: 420px;
    margin: auto;
    background: rgba(43, 0, 82, 0.45);
    border: 1px solid #7a00ff;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(122, 0, 255, 0.35);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# FIREBASE CONFIG (Web config is safe to keep here)
# -----------------------------------------------------------
firebaseConfig = {
  "apiKey": "AIzaSyDWT0YnQqdFC9KA-hLDdDSpvGrk3zLlmWo",
  "authDomain": "stock-sense-project.firebaseapp.com",
  "projectId": "stock-sense-project",
  "storageBucket": "stock-sense-project.firebasestorage.app",
  "messagingSenderId": "423833753864",
  "appId": "1:423833753864:web:a32af956e161bac98574bd",
  "measurementId": "G-YVW1MPQWLS",
  "databaseURL": "https://stock-sense-project-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth_fb = firebase.auth()

# -----------------------------------------------------------
# AUTH FUNCTIONS
# -----------------------------------------------------------
def login_user(email, password):
    try:
        auth_fb.sign_in_with_email_and_password(email, password)
        return True
    except:
        return False

def signup_user(email, password):
    try:
        auth_fb.create_user_with_email_and_password(email, password)
        return True
    except:
        return False

# -----------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------
st.markdown("<h1 class='title'>Stock Sense</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI Powered Stock Analysis Dashboard</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔐 Login", "🆕 Register"])

# -----------------------------------------------------------
# LOGIN TAB
# -----------------------------------------------------------
with tab1:
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    login_btn = st.button("Login")

    if login_btn:
        with st.spinner("Authenticating..."):
            success = login_user(email, password)
            time.sleep(1)

        if success:
            st.success("Login successful! Redirecting...")
            time.sleep(0.8)
            st.switch_page("pages/dashboard.py")
        else:
            st.error("Invalid email or password")

# -----------------------------------------------------------
# REGISTER TAB
# -----------------------------------------------------------
with tab2:
    email_r = st.text_input("Email (Register)")
    password_r = st.text_input("Password (Register)", type="password")

    reg_btn = st.button("Create Account")

    if reg_btn:
        with st.spinner("Registering..."):
            created = signup_user(email_r, password_r)
            time.sleep(1)

        if created:
            st.success("Account created! Please login.")
        else:
            st.error("Registration failed. Try another email.")
