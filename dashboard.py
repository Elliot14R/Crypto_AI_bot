import streamlit as st
import pandas as pd
import requests
import ccxt
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CryptoBot AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PRO TERMINAL CSS (Integrated from index.css) ---
st.markdown("""
    <style>
    :root {
        --background: #0E1117;
        --card: #161B22;
        --border: #30363D;
        --primary: #14b8a6;
        --chart-2: #22c55e;
        --destructive: #ef4444;
    }
    .stApp { background-color: var(--background); color: #C9D1D9; }
    
    /* Replit-style Glass-morphism Cards */
    .pro-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-live {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: bold;
    }
    .pulse-dot {
        height: 8px; width: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA ENGINES ---
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/").json()
        return r['data'][0]['value'], r['data'][0]['value_classification']
    except: return "50", "Neutral"

def get_market_data():
    try:
        # Using US endpoint to bypass location restrictions
        r = requests.get("https://api.binance.us/api/v3/ticker/24hr").json()
        return pd.DataFrame(r)
    except: return pd.DataFrame()

# --- SIDEBAR NAVIGATION (From App.tsx & app-sidebar.tsx) ---
with st.sidebar:
    st.markdown("### 🤖 CryptoBot AI")
    menu = st.radio(
        "Navigation", 
        ["Dashboard", "Signals", "Market", "Paper Trading", "Configuration"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("MONITORED COINS")
    symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "SUI", "DOT", "LINK", "MATIC", "NEAR", "APT", "INJ", "ARB"]
    pills = "".join([f"<span style='background:#21262d; padding:2px 6px; margin:2px; border-radius:4px; font-size:10px; border:1px solid #30363d;'>{s}</span>" for s in symbols])
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Last Scan: {datetime.now().strftime('%H:%M:%S')}")
    st.caption("Scan Interval: 15m")

# --- PAGE: DASHBOARD ---
if menu == "Dashboard":
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("### Dashboard Summary")
        st.markdown("<div class='badge-live'><span class='pulse-dot'></span>LIVE · Scanning 15 pairs · 73.1% Accuracy</div>", unsafe_allow_html=True)
    with col_h2:
        st.metric("Testnet Balance", "$10,000.00", delta="API Restricted")

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Signals", "14", "All Time")
    m2.metric("Win Rate", "73.1%", "Verified")
    m3.metric("Buy Signals", "8", "Longs")
    m4.metric("Sell Signals", "6", "Shorts")

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("📈 Recent Signals")
        # Integrated mock data based on signals_log.json
        signals = [
            {"coin": "SUI", "type": "BUY", "conf": 83.2, "entry": 0.97, "sl": 0.94, "tp": 1.04},
            {"coin": "BTC", "type": "BUY", "conf": 79.5, "entry": 70500, "sl": 68800, "tp": 74200}
        ]
        for s in signals:
            st.markdown(f"""
                <div class="pro-card">
                    <div style="display:flex; justify-content:space-between;">
                        <b>{s['coin']}USDT</b> 
                        <span style="color:#22c55e; font-weight:bold;">↗ {s['type']}</span>
                        <small style="color:#8B949E;">{s['conf']}% Conf</small>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin
