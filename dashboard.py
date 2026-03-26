import streamlit as st
import pandas as pd
import ccxt
import time
from datetime import datetime

# ─── PAGE CONFIGURATION ──────────────────────────────────────────
st.set_page_config(
    page_title="CryptoBot AI Dashboard",
    page_icon="🤖",
    layout="wide",f
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Dark theme overrides */
    .stApp { background-color: #0E1117; }
    
    /* Monitored Coins Pill Badges */
    .coin-badge {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
        border: 1px solid #334155;
    }
    
    /* Signal Card Styling */
    .signal-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .buy-badge { background-color: rgba(16, 185, 129, 0.2); color: #10B981; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;}
    .sell-badge { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;}
    .metric-value { font-size: 24px; font-weight: bold; color: #F8FAFC; }
    .metric-label { font-size: 13px; color: #94A3B8; }
    </style>
""", unsafe_allow_html=True)

# ─── EXCHANGE CONNECTION ─────────────────────────────────────────
@st.cache_resource(ttl=60)
def init_exchange():
    """Initialize Binance connection with a proxy/US-friendly bypass"""
    try:
        exchange = ccxt.binance({
            "apiKey": st.secrets["BINANCE_API_KEY"],
            "secret": st.secrets["BINANCE_SECRET"],
            "enableRateLimit": True,
        })
        
        # This tells CCXT to use the US-friendly endpoint for public data
        # while keeping your Testnet credentials for the private account data
        exchange.urls['api']['public'] = 'https://api.binance.us/api/v3'
        exchange.set_sandbox_mode(True)
        
        return exchange
    except Exception as e:
        st.error(f"Failed to initialize exchange: {e}")
        return None

exchange = init_exchange()

# ─── SIDEBAR NAVIGATION & COINS ──────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    menu = st.radio(
        "Navigation Menu", # Required label to prevent crash
        ["Dashboard", "Signals", "Market", "Paper Trading", "Configuration"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<span style='color: #94A3B8; font-size: 14px;'>Monitored Coins</span>", unsafe_allow_html=True)
    
    symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "SUI", "DOT", "LINK", "MATIC", "NEAR", "APT", "ARB"]
    
    pills_html = "".join([f"<div class='coin-badge'>{sym}</div>" for sym in symbols])
    st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)

# ─── MAIN DASHBOARD CONTENT ──────────────────────────────────────
if menu == "Dashboard":
    
    # Header Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🤖 CryptoBot AI Dashboard")
        st.markdown("🟢 **LIVE** &nbsp; | &nbsp; Scanning 14 pairs &nbsp;•&nbsp; 15 min interval &nbsp;•&nbsp; 73.1% accuracy")
    
    with col2:
        balance_usdt = 0.00
        if exchange:
            try:
                bal = exchange.fetch_balance()
                balance_usdt = float(bal.get('USDT', {}).get('free', 0.0))
            except Exception as e:
                st.error(f"Binance API Error: {e}")
        
        st.metric("Testnet Balance", f"${balance_usdt:,.2f}")

    st.markdown("---")
    
    # Top Metrics Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Signals (All Time)", "14")
    with m2:
        st.metric("Buy Signals", "8", delta="Long positions active", delta_color="normal")
    with m3:
        st.metric("Sell Signals", "6", delta="Short positions active", delta_color="inverse")

    # Recent Signals Section
    st.markdown("<br><h4>📈 Recent Signals</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="signal-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="font-size: 18px; font-weight: bold; color: white;">
                BTC &nbsp;<span class="buy-badge">↗ BUY</span> &nbsp;<span style="color: #94A3B8; font-size: 14px;">78.4% Confidence</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div>
                <div class="metric-label">Entry Price</div>
                <div class="metric-value">83,420.50</div>
            </div>
            <div>
                <div class="metric-label">Stop Loss</div>
                <div class="metric-value" style="color: #EF4444;">81,850.00 <span style="font-size: 14px;">(-1.9%)</span></div>
            </div>
            <div>
                <div class="metric-label">Target (TP1)</div>
                <div class="metric-value" style="color: #10B981;">84,990.00 <span style="font-size: 14px;">(+1.9%)</span></div>
            </div>
        </div>
        <div style="color: #64748B; font-size: 13px;">
            🕒 25 minutes ago &nbsp; • &nbsp; ADX: 28 &nbsp; • &nbsp; RSI: 52
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── OTHER TABS ──────────────────────────────────────────────────
elif menu == "Paper Trading":
    st.markdown("## 📜 Active Testnet Orders")
    if exchange:
        with st.spinner("Fetching orders from Binance Testnet..."):
            try:
                orders = exchange.fetch_open_orders()
                if not orders:
                    st.info("No open orders. Waiting for the bot to fire a signal.")
                else:
                    st.json(orders)
            except Exception as e:
                st.error(f"Error fetching orders: {e}")
    else:
        st.warning("Exchange not initialized.")

elif menu == "Configuration":
    st.markdown("## ⚙️ Bot Configuration")
    st.code("""
    MIN_CONFIDENCE    = 75
    MIN_ADX           = 25
    RISK_PER_TRADE    = 0.01 (1%)
    MAX_OPEN_TRADES   = 3
    TIMEFRAME_ENTRY   = '15m'
    """, language="python")

else:
    st.markdown(f"## {menu}")
    st.info("This section is under construction.")
