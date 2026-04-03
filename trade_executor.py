# trade_executor.py — FINAL CLEAN VERSION (NO 451 ERROR)

import os, time, logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from persistence import save_json, load_json
from binance_client import BinanceTestnet

load_dotenv()

# ================== CONFIG ==================
BALANCE_FILE = "balance.json"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ================== INIT ==================

def init_exchange():
    key = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_SECRET", "")

    if not key or not secret:
        raise ValueError("Missing API keys")

    return BinanceTestnet(key, secret)

# ================== BALANCE ==================

def fetch_and_save_balance():
    try:
        ex = init_exchange()

        balances = ex.get_balance()

        usdt = balances.get("USDT", 0.0)

        assets = []
        for asset, amount in balances.items():
            if amount > 0:
                assets.append({
                    "asset": asset,
                    "free": round(amount, 6),
                    "locked": 0,
                    "total": round(amount, 6)
                })

        save_json(BALANCE_FILE, {
            "usdt": round(usdt, 2),
            "assets": assets,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        log.info(f"✅ Balance fetched: {usdt} USDT | {len(assets)} assets")

        return usdt

    except Exception as e:
        log.error(f"❌ Balance error: {e}")

        save_json(BALANCE_FILE, {
            "usdt": 0,
            "assets": [],
            "error": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        return 0

# ================== TEST RUN ==================

if __name__ == "__main__":
    log.info("Starting bot...")

    balance = fetch_and_save_balance()

    if balance <= 0:
        log.warning("⚠️ Balance is zero or API blocked")
    else:
        log.info("🎉 SUCCESS — Bot connected to Binance")
