# # paper_trader.py — FINAL STABLE VERSION

# import json
# import logging
# import requests
# import time
# from datetime import datetime, timezone
# from pathlib import Path

# log = logging.getLogger(__name__)

# BALANCE_FILE  = "balance.json"
# TRADES_FILE   = "trades.json"


# # ── 🔥 FIXED: Robust price fetch ────────────────

# def get_live_price(symbol: str) -> float:
#     """Robust price fetch with fallback sources"""

#     # 1️⃣ Primary: Binance
#     try:
#         r = requests.get(
#             "https://api.binance.com/api/v3/ticker/price",
#             params={"symbol": symbol},
#             timeout=10
#         )
#         if r.status_code == 200:
#             data = r.json()
#             price = float(data.get("price", 0))
#             if price > 0:
#                 return price
#     except Exception as e:
#         log.warning(f"Binance price error {symbol}: {e}")

#     # 2️⃣ 🔥 Fallback: Binance data API (different endpoint)
#     try:
#         r = requests.get(
#             "https://data-api.binance.vision/api/v3/ticker/price",
#             params={"symbol": symbol},
#             timeout=10
#         )
#         if r.status_code == 200:
#             data = r.json()
#             price = float(data.get("price", 0))
#             if price > 0:
#                 return price
#     except Exception as e:
#         log.warning(f"Fallback price error {symbol}: {e}")

#     # 3️⃣ Final fail
#     log.warning(f"❌ All price sources failed for {symbol}")
#     return 0.0


# def get_live_prices_bulk(symbols: list) -> dict:
#     try:
#         r = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=8)
#         if r.ok:
#             return {item["symbol"]: float(item["price"])
#                     for item in r.json()
#                     if item["symbol"] in symbols}
#     except Exception as e:
#         log.warning(f"Bulk price fetch: {e}")
#     return {}


# # ── JSON helpers ───────────────────────────────

# def load_json(path, default):
#     try:
#         if Path(path).exists():
#             with open(path) as f:
#                 return json.load(f)
#     except Exception:
#         pass
#     return default


# def save_json_file(path, data):
#     try:
#         import os
#         tmp = str(path) + ".tmp"
#         with open(tmp, "w") as f:
#             json.dump(data, f, indent=2, default=str)
#         os.replace(tmp, path)
#     except Exception as e:
#         log.error(f"save_json {path}: {e}")


# # ── Paper Trader ───────────────────────────────

# class PaperTrader:

#     STARTING_BALANCE = 10000.0
    
#     def test_connection(self) -> bool:
#         """Always succeeds for paper trading"""
#         log.info("  ✅ Paper trading mode — no exchange connection needed")
#         return True

#     def __init__(self):
#         self._ensure_balance()

#     def _ensure_balance(self):
#         bal = load_json(BALANCE_FILE, {})
#         if not bal or bal.get("usdt") in (None, 0):
#             self._save_balance(self.STARTING_BALANCE)
#             log.info(f"  ✅ Initialized balance: {self.STARTING_BALANCE} USDT")

#     def _save_balance(self, usdt: float, assets: list = None):
#         save_json_file(BALANCE_FILE, {
#             "usdt": round(usdt, 4),
#             "assets": assets or [{"asset":"USDT","free":str(round(usdt,4)),"total":str(round(usdt,4))}],
#             "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
#             "mode": "paper_trading",
#         })

#     def get_usdt_balance(self) -> float:
#         bal = load_json(BALANCE_FILE, {})
#         trades = load_json(TRADES_FILE, {})

#         locked = sum(
#             t.get("qty", 0) * t.get("entry", 0)
#             for t in trades.values()
#             if not t.get("closed")
#         )

#         free_usdt = float(bal.get("usdt", self.STARTING_BALANCE))
#         return max(0.0, free_usdt - locked)

#     def get_balance(self) -> dict:
#         bal = load_json(BALANCE_FILE, {})
#         return {"USDT": float(bal.get("usdt", self.STARTING_BALANCE))}

#     def save_balance_snapshot(self):
#         trades = load_json(TRADES_FILE, {})
#         bal = load_json(BALANCE_FILE, {})
#         base_usdt = float(bal.get("usdt", self.STARTING_BALANCE))

#         symbols = [s for s in trades if not trades[s].get("closed")]
#         prices = get_live_prices_bulk(symbols) if symbols else {}
#         upnl = 0.0

#         for sym, t in trades.items():
#             if t.get("closed"):
#                 continue

#             live = prices.get(sym)
#             if live and t.get("entry") and t.get("qty"):
#                 if t["signal"] == "BUY":
#                     upnl += (live - t["entry"]) * t["qty"]
#                 else:
#                     upnl += (t["entry"] - live) * t["qty"]

#         equity = base_usdt + upnl

#         save_json_file(BALANCE_FILE, {
#             "usdt": round(base_usdt, 4),
#             "equity": round(equity, 4),
#             "unrealised": round(upnl, 4),
#             "assets": [
#                 {
#                     "asset": "USDT",
#                     "free": str(round(base_usdt, 4)),
#                     "total": str(round(equity, 4))
#                 }
#             ],
#             "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
#             "mode": "paper_trading",
#             "open_trades": len(symbols)
#         })

#         log.info(f"  ✅ balance.json: {base_usdt:.2f} | PnL: {upnl:+.2f}")

#     # ── Order simulation ─────────────────────────

#     def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:

#         # 🔥 FIX: safe price check
#         live_price = get_live_price(symbol)
#         if live_price <= 0:
#             raise Exception(f"Cannot get live price for {symbol}")

#         slippage = 1.0005 if side.upper() == "BUY" else 0.9995
#         fill_price = live_price * slippage

#         order_id = f"paper_{symbol}_{int(datetime.now(timezone.utc).timestamp())}"

#         log.info(f"  📝 PAPER {side} {quantity:.6f} {symbol} @ {fill_price:.4f}")

#         return {
#             "orderId": order_id,
#             "symbol": symbol,
#             "side": side,
#             "status": "FILLED",
#             "price": str(fill_price),
#             "paper_fill": fill_price,
#         }

#     def place_limit_order(self, symbol: str, side: str, quantity: float,
#                       price: float, stop_price: float = None) -> dict:
#         """Simulate limit / stop orders for TP & SL"""

#         order_id = f"paper_{symbol}_{side}_{int(datetime.now(timezone.utc).timestamp())}"

#         order_type = "STOP_LOSS" if stop_price else "LIMIT"

#         log.info(f"  📝 PAPER {order_type} {side} {quantity:.6f} {symbol} @ {price:.4f}")

#         return {
#             "orderId": order_id,
#             "symbol": symbol,
#             "side": side.upper(),
#             "type": order_type,
#             "origQty": str(quantity),
#             "price": str(price),
#             "stopPrice": str(stop_price or price),
#             "status": "NEW",
#         }

#     def update_balance_after_close(self, pnl: float):
#         bal = load_json(BALANCE_FILE, {})
#         usdt = float(bal.get("usdt", self.STARTING_BALANCE))
#         new_balance = usdt + pnl
#         self._save_balance(new_balance)

#         log.info(f"  💰 Balance updated: {usdt} → {new_balance}")
