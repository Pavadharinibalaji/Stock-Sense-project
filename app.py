import streamlit as st
import pyrebase
import time
from st_pages import hide_pages
import os


# Load Finnhub key from Render environment variable
finnhub_api = os.getenv("FINNHUB_API_KEY")


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Stock Sense - Login",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide sidebar
hide_pages(["dashboard"])

st.markdown("""
<style>
/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a0030 0%, #120022 40%, #0d001a 100%);
    color: #00ffff !important;
}

/* TITLE */
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

/* GLASS CARD */
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

/* INPUT FIELDS */
.stTextInput>div>div>input {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #000000 !important;
    border-radius: 10px !important;
    border: 1px solid #7a00ff !important;
    font-size: 1rem;
}

label {
    font-weight: 600 !important;
    color: #00ffff !important;
    font-family: "Segoe UI", sans-serif;
}
/* MAKE BUTTON CONTAINER FULL WIDTH */
button[kind="primary"], .stButton, .stButton > button {
    width: 100% !important;
}
/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #00ffff, #7a00ff);
    color: #1e0033 !important;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.6rem;
    box-shadow: 0px 0px 20px #00eaff;
    transition: 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #7a00ff, #00ffff);
    box-shadow: 0px 0px 28px #00ffff;
    transform: scale(1.03);
}

/* TABS */
.stTabs [role="tab"] {
    background: rgba(43, 0, 82, 0.5);
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# FIREBASE CONFIG
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

    st.markdown('<div style="width:100%;">', unsafe_allow_html=True)
    login_btn = st.button("Login")
    st.markdown('</div>', unsafe_allow_html=True)

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
# SIGNUP TAB
# -----------------------------------------------------------
with tab2:

    email_r = st.text_input("Email (Register)")
    password_r = st.text_input("Password (Register)", type="password")

    st.markdown('<div style="width:100%;">', unsafe_allow_html=True)
    reg_btn = st.button("Create Account")
    st.markdown('</div>', unsafe_allow_html=True)

    if reg_btn:

        with st.spinner("Registering..."):
            created = signup_user(email_r, password_r)
            time.sleep(1)

        if created:
            st.success("Account created! Please login.")
        else:
            st.error("Registration failed. Try another email.")

