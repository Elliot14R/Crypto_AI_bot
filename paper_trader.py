# paper_trader.py — FINAL FIXED VERSION (GITHUB SYNC ENABLED)

import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

# ✅ IMPORTANT: use persistence (GitHub sync)
from persistence import save_json

log = logging.getLogger(__name__)

BALANCE_FILE = "balance.json"
TRADES_FILE  = "trades.json"


# ── Public price fetch ─────────────────────────

def get_live_price(symbol: str) -> float:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=8
        )
        if r.ok:
            return float(r.json()["price"])
    except Exception as e:
        log.warning(f"Price fetch {symbol}: {e}")
    return 0.0


def get_live_prices_bulk(symbols: list) -> dict:
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=8)
        if r.ok:
            return {
                item["symbol"]: float(item["price"])
                for item in r.json()
                if item["symbol"] in symbols
            }
    except Exception as e:
        log.warning(f"Bulk price fetch: {e}")
    return {}


# ── JSON helpers (read only) ───────────────────

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


# ── Paper Trader ───────────────────────────────

class PaperTrader:

    STARTING_BALANCE = 10000.0

    def __init__(self):
        self._ensure_balance()

    # 🔥 FIXED: force initialize if 0
    def _ensure_balance(self):
        bal = load_json(BALANCE_FILE, {})

        if not bal or bal.get("usdt") in (None, 0):
            self._save_balance(self.STARTING_BALANCE)
            log.info(f"  ✅ Initialized balance: {self.STARTING_BALANCE} USDT")

    # 🔥 FIXED: uses GitHub persistence
    def _save_balance(self, usdt: float):
        save_json(BALANCE_FILE, {
            "usdt": round(usdt, 4),
            "assets": [
                {
                    "asset": "USDT",
                    "free": str(round(usdt, 4)),
                    "total": str(round(usdt, 4))
                }
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "paper_trading"
        })

    def get_usdt_balance(self) -> float:
        bal = load_json(BALANCE_FILE, {})
        trades = load_json(TRADES_FILE, {})

        locked = sum(
            t.get("qty", 0) * t.get("entry", 0)
            for t in trades.values()
            if not t.get("closed")
        )

        free_usdt = float(bal.get("usdt", self.STARTING_BALANCE))
        return max(0.0, free_usdt - locked)

    def get_balance(self) -> dict:
        bal = load_json(BALANCE_FILE, {})
        return {"USDT": float(bal.get("usdt", self.STARTING_BALANCE))}

    # 🔥 FIXED: GitHub sync here also
    def save_balance_snapshot(self):
        trades = load_json(TRADES_FILE, {})
        bal = load_json(BALANCE_FILE, {})

        base_usdt = float(bal.get("usdt", self.STARTING_BALANCE))

        symbols = [s for s in trades if not trades[s].get("closed")]
        prices = get_live_prices_bulk(symbols) if symbols else {}

        upnl = 0.0

        for sym, t in trades.items():
            if t.get("closed"):
                continue

            live = prices.get(sym)
            if live and t.get("entry") and t.get("qty"):
                if t["signal"] == "BUY":
                    upnl += (live - t["entry"]) * t["qty"]
                else:
                    upnl += (t["entry"] - live) * t["qty"]

        equity = base_usdt + upnl

        save_json(BALANCE_FILE, {
            "usdt": round(base_usdt, 4),
            "equity": round(equity, 4),
            "unrealised": round(upnl, 4),
            "assets": [
                {
                    "asset": "USDT",
                    "free": str(round(base_usdt, 4)),
                    "total": str(round(equity, 4))
                }
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "paper_trading",
            "open_trades": len(symbols)
        })

        log.info(f"  ✅ balance.json: {base_usdt:.2f} USDT | unrealised: {upnl:+.2f} | equity: {equity:.2f}")
        return base_usdt

    # ── Orders ─────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        live_price = get_live_price(symbol)

        if live_price <= 0:
            raise Exception(f"Price fetch failed for {symbol}")

        slippage = 1.0005 if side.upper() == "BUY" else 0.9995
        fill_price = live_price * slippage

        order_id = f"paper_{symbol}_{int(datetime.now(timezone.utc).timestamp())}"

        log.info(f"  📝 PAPER {side} {quantity} {symbol} @ {fill_price:.4f}")

        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "status": "FILLED",
            "price": str(fill_price),
            "paper_fill": fill_price,
        }

    def place_limit_order(self, symbol, side, quantity, price, stop_price=None):
        order_id = f"paper_{symbol}_{side}_{int(datetime.now(timezone.utc).timestamp())}"
        return {
            "orderId": order_id,
            "symbol": symbol,
            "price": str(price),
            "status": "NEW"
        }

    def get_order(self, symbol, order_id):
        return {"status": "NEW"}

    def update_balance_after_close(self, pnl: float):
        bal = load_json(BALANCE_FILE, {})
        usdt = float(bal.get("usdt", self.STARTING_BALANCE))

        new_balance = usdt + pnl
        self._save_balance(new_balance)

        log.info(f"  💰 Balance updated: {usdt} → {new_balance} (PnL {pnl})")

    def test_connection(self):
        log.info("  ✅ Paper trading mode active")
        return True
