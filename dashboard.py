# dashboard_api.py
# Flask backend — serves live trade data + triggers manual scans
# Run locally: python dashboard_api.py
# Or deploy free on Render.com / Railway.app

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

app = Flask(__name__, static_folder="dashboard_static")
CORS(app)   # allow dashboard to call this API from any origin

TRADES_FILE   = "trades.json"
LOG_FILE      = "bot.log"
HISTORY_FILE  = "trade_history.json"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def load_trade_history():
    return load_json(HISTORY_FILE, [])


def save_trade_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


def get_live_prices(symbols):
    """Fetch current prices from Binance public API."""
    prices = {}
    try:
        import requests as req
        for sym in symbols:
            try:
                r = req.get(
                    f"https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": sym}, timeout=5
                )
                prices[sym] = float(r.json()["price"])
            except Exception:
                prices[sym] = None
    except Exception:
        pass
    return prices


def enrich_trades(trades: dict, prices: dict) -> list:
    """Add live PnL and price data to open trades."""
    result = []
    for sym, t in trades.items():
        entry      = t.get("entry", 0)
        qty        = t.get("qty", 0)
        signal     = t.get("signal", "BUY")
        live_price = prices.get(sym)

        if live_price and entry and qty:
            if signal == "BUY":
                unrealised_pnl = round((live_price - entry) * qty, 4)
                pnl_pct        = round((live_price - entry) / entry * 100, 2)
            else:
                unrealised_pnl = round((entry - live_price) * qty, 4)
                pnl_pct        = round((entry - live_price) / entry * 100, 2)
        else:
            unrealised_pnl = 0
            pnl_pct        = 0

        sl   = t.get("stop", 0)
        tp1  = t.get("tp1", 0)
        tp2  = t.get("tp2", 0)
        dec  = 4 if entry < 10 else 2

        # Progress toward TP (0–100%)
        if signal == "BUY" and tp2 > entry > sl:
            progress = round(max(0, min(100, (live_price - entry) / (tp2 - entry) * 100)), 1) if live_price else 0
        elif signal == "SELL" and tp2 < entry < sl:
            progress = round(max(0, min(100, (entry - live_price) / (entry - tp2) * 100)), 1) if live_price else 0
        else:
            progress = 0

        result.append({
            **t,
            "live_price":     live_price,
            "unrealised_pnl": unrealised_pnl,
            "pnl_pct":        pnl_pct,
            "progress":       progress,
            "status":         "TP1 hit" if t.get("tp1_hit") else "Open",
        })
    return result


# ── API Routes ───────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """Dashboard health check + summary stats."""
    trades  = load_json(TRADES_FILE, {})
    history = load_trade_history()

    wins    = [h for h in history if h.get("pnl", 0) > 0]
    losses  = [h for h in history if h.get("pnl", 0) <= 0]
    total   = len(history)
    win_rate = round(len(wins) / total * 100, 1) if total else 0
    total_pnl = round(sum(h.get("pnl", 0) for h in history), 4)

    return jsonify({
        "ok":           True,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "open_trades":  len(trades),
        "max_trades":   3,
        "total_closed": total,
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     win_rate,
        "total_pnl":    total_pnl,
        "model_acc":    73.1,
    })


@app.route("/api/trades/open")
def api_open_trades():
    """Return all open trades with live prices + unrealised PnL."""
    trades  = load_json(TRADES_FILE, {})
    symbols = list(trades.keys())
    prices  = get_live_prices(symbols)
    return jsonify(enrich_trades(trades, prices))


@app.route("/api/trades/history")
def api_trade_history():
    """Return closed trade history."""
    history = load_trade_history()
    return jsonify(history[-50:])   # last 50 trades


@app.route("/api/log")
def api_log():
    """Return last 100 lines of bot.log."""
    try:
        if Path(LOG_FILE).exists():
            lines = Path(LOG_FILE).read_text().splitlines()
            return jsonify({"lines": lines[-100:]})
    except Exception:
        pass
    return jsonify({"lines": ["Log file not found"]})


@app.route("/api/scan", methods=["POST"])
def api_trigger_scan():
    """Manually trigger a scan (runs trade_executor in background)."""
    def run_scan():
        try:
            import subprocess
            subprocess.run(["python", "trade_executor.py"], timeout=300)
        except Exception as e:
            log.error(f"Manual scan error: {e}")

    t = threading.Thread(target=run_scan, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Scan started"})


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    """Manually close a trade by symbol."""
    data   = request.get_json()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400

    trades = load_json(TRADES_FILE, {})
    if symbol not in trades:
        return jsonify({"ok": False, "error": "Trade not found"}), 404

    try:
        from trade_executor import init_exchange
        exchange = init_exchange()
        trade    = trades[symbol]
        order_ids = trade.get("order_ids", {})

        # Cancel all open orders for this symbol
        for key, oid in order_ids.items():
            try:
                exchange.cancel_order(oid, symbol)
            except Exception:
                pass

        # Market close
        close_side = "sell" if trade["signal"] == "BUY" else "buy"
        exchange.create_order(symbol, "market", close_side, trade["qty"])

        # Record in history
        prices  = get_live_prices([symbol])
        close_p = prices.get(symbol, trade["entry"])
        entry   = trade["entry"]
        qty     = trade["qty"]
        if trade["signal"] == "BUY":
            pnl = round((close_p - entry) * qty, 4)
        else:
            pnl = round((entry - close_p) * qty, 4)

        history = load_trade_history()
        history.append({
            **trade,
            "close_price": close_p,
            "pnl":         pnl,
            "closed_at":   datetime.now(timezone.utc).isoformat(),
            "close_reason": "Manual close",
        })
        save_trade_history(history)

        trades.pop(symbol)
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)

        return jsonify({"ok": True, "pnl": pnl})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/")
def serve_dashboard():
    return send_from_directory("dashboard_static", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("dashboard_static", path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard API running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
