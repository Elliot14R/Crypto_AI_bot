import streamlit as st
import pandas as pd
import ccxt
import time
from datetime import datetime

# ─── PAGE CONFIGURATION ──────────────────────────────────────────
st.set_page_config(
    page_title="CryptoBot AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS (To match your Replit design) ────────────────────
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
@st.cache_resource(ttl=60) # Caches connection for 60 seconds to prevent API spam
def init_exchange():
    """Initialize Binance Testnet using Streamlit Secrets"""
    try:
        # Streamlit uses st.secrets instead of .env for cloud deployments
        exchange = ccxt.binance({
            "apiKey": st.secrets["BINANCE_API_KEY"],
            "secret": st.secrets["BINANCE_SECRET"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
        exchange.set_sandbox_mode(True)
        return exchange
    except Exception as e:
        return None

exchange = init_exchange()

# ─── SIDEBAR NAVIGATION & COINS ──────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    menu = st.radio(
        "Navigation Menu",
        ["Dashboard", "Signals", "Market", "Paper Trading", "Configuration"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<span style='color: #94A3B8; font-size: 14px;'>Monitored Coins</span>", unsafe_allow_html=True)
    
    # Matching your config.py symbols
    symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "SUI", "DOT", "LINK", "MATIC", "NEAR", "APT", "ARB"]
    
    # Generate HTML for the coin pills
    pills_html = ""
    for sym in symbols:
        pills_html += f"<div class='coin-badge'>{sym}</div>"
    st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)

# ─── MAIN DASHBOARD CONTENT ──────────────────────────────────────
if menu == "Dashboard":
    
    # Header Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 🤖 CryptoBot AI Dashboard")
        st.markdown("🟢 **LIVE** &nbsp; | &nbsp; Scanning 14 pairs &nbsp;•&nbsp; 15 min interval &nbsp;•&nbsp; 73.1% accuracy")
    with col2:
        # Fetch Live Balance
        balance_usdt = 0.00
        if exchange:
            try:
                bal = exchange.fetch_balance()
                balance_usdt = float(bal['USDT']['free'])
            except:
                pass
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

    # Recent Signals Section (Replicating your image)
    st.markdown("<br><h4>📈 Recent Signals</h4>", unsafe_allow_html=True)
    
    # Example Trade Card (This matches your Replit screenshot exactly)
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

# ─── OTHER TABS (Placeholders for future expansion) ──────────────
elif menu == "Paper Trading":
    st.markdown("## 📜 Active Testnet Orders")
    if exchange:
        with st.spinner("Fetching orders from Binance Testnet..."):
            try:
                orders = exchange.fetch_open_orders()
                if not orders:
                    st.info("No open orders right now. Waiting for the bot to fire a signal.")
                else:
                    st.write(orders) # Will display the raw JSON of open orders for now
            except Exception as e:
                st.error(f"Error fetching orders: {e}")
    else:
        st.warning("Please configure your API keys in Streamlit Secrets to view live orders.")

elif menu == "Configuration":
    st.markdown("## ⚙️ Bot Configuration")
    st.code("""
    # Current active settings
    MIN_CONFIDENCE    = 75
    MIN_ADX           = 25
    RISK_PER_TRADE    = 0.01 (1%)
    MAX_OPEN_TRADES   = 3
    TIMEFRAME_ENTRY   = '15m'
    """, language="python")

else:
    st.markdown(f"## {menu}")
    st.info("This section is under construction.")
