# dashboard_api.py - Complete fixed version
import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)

# ── THE FIX: Tell Python to save logs to 'bot.log' AND the console ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

TRADES_FILE  = "trades.json"
LOG_FILE     = "bot.log"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"

# ── Persistence layer ─────────────────────────────────────────
def load_json(path, default):
    try:
        from persistence import load_from_github
        data = load_from_github(Path(path).name, None)
        if data is not None:
            return data
    except Exception:
        pass
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    try:
        from persistence import save_json as persist_save
        persist_save(path, data)
        return
    except Exception:
        pass
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Save failed {path}: {e}")

def restore_on_startup():
    try:
        from persistence import pull_all_from_github
        count = pull_all_from_github()
        log.info(f"Dashboard startup: restored {count} files from GitHub")
    except Exception as e:
        log.warning(f"Startup restore failed: {e}")

# ── Live price fetching ──────────────────────────────────────
_price_cache   = {}
_price_cache_t = 0
PRICE_CACHE_TTL = 10  # seconds

def get_live_prices(symbols):
    global _price_cache, _price_cache_t
    now = time.time()

    if now - _price_cache_t < PRICE_CACHE_TTL and _price_cache:
        return {s: _price_cache.get(s) for s in symbols}

    prices = {}
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
        data = r.json()
        for item in data:
            _price_cache[item["symbol"]] = {
                "price":      float(item["lastPrice"]),
                "change_pct": float(item["priceChangePercent"]),
                "high":       float(item["highPrice"]),
                "low":        float(item["lowPrice"]),
                "volume":     float(item["quoteVolume"]),
            }
        _price_cache_t = now
    except Exception as e:
        log.warning(f"Price fetch error: {e}")

    return {s: _price_cache.get(s) for s in symbols}

def enrich_trades(trades, prices):
    result = []
    for sym, t in trades.items():
        entry  = t.get("entry", 0)
        qty    = t.get("qty",   0)
        signal = t.get("signal", "BUY")
        pd     = prices.get(sym) or {}
        live   = pd.get("price") if pd else None

        if live and entry and qty:
            upnl = round((live - entry) * qty if signal == "BUY" else (entry - live) * qty, 4)
            pct  = round((live - entry) / entry * 100 if signal == "BUY" else (entry - live) / entry * 100, 2)
        else:
            upnl, pct = 0, 0

        tp2 = t.get("tp2", 0)
        sl  = t.get("stop", 0)
        if signal == "BUY" and tp2 > entry > sl and live:
            prog = round(max(0, min(100, (live - entry) / (tp2 - entry) * 100)), 1)
        elif signal == "SELL" and tp2 < entry < sl and live:
            prog = round(max(0, min(100, (entry - live) / (entry - tp2) * 100)), 1)
        else:
            prog = 0

        result.append({
            **t,
            "live_price":     live,
            "unrealised_pnl": upnl,
            "pnl_pct":        pct,
            "progress":       prog,
            "status":         "TP1 hit" if t.get("tp1_hit") else "Open",
        })
    return result

# ── Telegram Listener ─────────────────────────────────────────
def telegram_listener():
    """Runs in the background to listen for your Telegram commands."""
    token = os.getenv("TELEGRAM_TOKEN")
    if not token: return
    
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            params = {"timeout": 10, "allowed_updates": ["message"]}
            if offset: params["offset"] = offset
            
            r = requests.get(url, params=params, timeout=15)
            if r.ok:
                data = r.json()
                for item in data.get("result", []):
                    offset = item["update_id"] + 1
                    msg = item.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = msg.get("chat", {}).get("id")
                    
                    if text == "/status":
                        trades = load_json(TRADES_FILE, {})
                        if not trades:
                            reply = "📡 *Status:* No open trades."
                        else:
                            reply = f"📡 *Status:* {len(trades)} Open Trades\n"
                            for sym, t in trades.items():
                                reply += f"• {sym} ({t['signal']}) @ {t['entry']}\n"
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                                      data={"chat_id": chat_id, "text": reply, "parse_mode": "Markdown"})
                        
                    elif text.startswith("/close"):
                        parts = text.split(" ")
                        if len(parts) > 1:
                            symbol = parts[1].upper()
                            if not symbol.endswith("USDT"): symbol += "USDT"
                            port = int(os.getenv("PORT", 5000))
                            requests.post(f"http://localhost:{port}/api/close_trade", json={"symbol": symbol})
                            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                                          data={"chat_id": chat_id, "text": f"⚙️ Attempting to close {symbol}..."})
        except Exception:
            pass
        time.sleep(3)


# ── Automated Scanner ─────────────────────────────────────────
def automated_scanner():
    """Runs the trading algorithm automatically every 15 minutes aligned to the clock."""
    while True:
        try:
            # Calculate seconds until the next :00, :15, :30, or :45 mark
            now = datetime.now(timezone.utc)
            minutes_to_wait = 15 - (now.minute % 15)
            seconds_to_wait = (minutes_to_wait * 60) - now.second
            
            # Sleep until the exact 15-minute mark
            time.sleep(seconds_to_wait)
            
            log.info(f"⏰ Auto-Scheduler: Triggering 15-min market scan...")
            import subprocess
            subprocess.run(["python", "trade_executor.py"], timeout=300)
        except Exception as e:
            log.error(f"Auto-scan scheduler failed: {e}")
            time.sleep(60)


# ── API routes ────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    trades  = load_json(TRADES_FILE,  {})
    history = load_json(HISTORY_FILE, [])
    signals = load_json(SIGNALS_FILE, [])

    wins    = [h for h in history if (h.get("pnl", 0) or 0) > 0]
    losses  = [h for h in history if (h.get("pnl", 0) or 0) <= 0]
    total   = len(history)
    wr      = round(len(wins) / total * 100, 1) if total else 0
    totpnl  = round(sum(h.get("pnl", 0) or 0 for h in history), 4)

    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_sigs = [s for s in signals if (s.get("generated_at") or "")[:10] == today]

    try:
        with open("model_performance.json") as f:
            perf = json.load(f)
        model_acc = round(perf.get("test_accuracy", 0.731) * 100, 1)
    except Exception:
        model_acc = 73.1

    mode_data = load_json("scan_mode.json", {})

    return jsonify({
        "ok":            True,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "open_trades":   len(trades),
        "max_trades":    3,
        "total_closed":  total,
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      wr,
        "total_pnl":     totpnl,
        "model_acc":     model_acc,
        "today_signals": len(today_sigs),
        "today_buy":     len([s for s in today_sigs if s.get("signal") == "BUY"]),
        "today_sell":    len([s for s in today_sigs if s.get("signal") == "SELL"]),
        "scan_mode":     mode_data.get("mode", "active"),
    })


@app.route("/api/balance")
def api_balance():
    try:
        from trade_executor import init_exchange
        ex = init_exchange()
        b  = ex.fetch_balance()

        free_usdt  = float(b.get("USDT", {}).get("free",  0) or 0)
        total_usdt = float(b.get("USDT", {}).get("total", 0) or 0)

        trades = load_json(TRADES_FILE, {})
        syms   = list(trades.keys())
        prices = get_live_prices(syms) if syms else {}
        upnl   = sum(
            (prices.get(sym, {}).get("price", t["entry"]) - t["entry"]) * t["qty"]
            if t["signal"] == "BUY"
            else (t["entry"] - prices.get(sym, {}).get("price", t["entry"])) * t["qty"]
            for sym, t in trades.items()
            if t.get("entry") and t.get("qty")
        )

        assets = [
            {"asset": a, "free": str(b[a]["free"]), "total": str(b[a]["total"])}
            for a in b if isinstance(b[a], dict) and float(b[a].get("free", 0) or 0) > 0
            and a not in ("info", "free", "used", "total", "timestamp", "datetime")
        ]

        return jsonify({
            "ok":          True,
            "usdt":        free_usdt,
            "total_usdt":  total_usdt,
            "equity":      round(free_usdt + upnl, 2),
            "unrealised":  round(upnl, 4),
            "assets":      assets,
            "note":        "Equity = Free USDT + Open Trade P&L",
        })
    except Exception as e:
        return jsonify({"ok": False, "usdt": None, "equity": None, "unrealised": 0, "error": str(e)})


@app.route("/api/trades/open")
def api_open():
    trades = load_json(TRADES_FILE, {})
    syms   = list(trades.keys())
    prices = get_live_prices(syms) if syms else {}
    return jsonify(enrich_trades(trades, prices))


@app.route("/api/trades/history")
def api_history():
    return jsonify(load_json(HISTORY_FILE, [])[-100:])


@app.route("/api/signals")
def api_signals():
    signals  = load_json(SIGNALS_FILE, [])
    symbol   = request.args.get("symbol")
    sig_type = request.args.get("type")
    rejected = request.args.get("rejected") 

    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol]
    if sig_type:
        signals = [s for s in signals if s.get("signal") == sig_type.upper()]
    if rejected == "false":
        signals = [s for s in signals if not s.get("rejected")]
    elif rejected == "true":
        signals = [s for s in signals if s.get("rejected")]

    return jsonify(list(reversed(signals[-200:])))


@app.route("/api/market/overview")
def api_market_overview():
    from config import SYMBOLS
    prices  = get_live_prices(SYMBOLS)
    changes = [v.get("change_pct", 0) for v in prices.values() if v]

    bullish = sum(1 for c in changes if c > 0)
    bearish = sum(1 for c in changes if c < 0)
    avg     = round(sum(changes) / len(changes), 2) if changes else 0

    fg_score, fg_label = 50, "Neutral"
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        d = r.json()
        fg_score = int(d["data"][0]["value"])
        fg_label = d["data"][0]["value_classification"]
    except Exception:
        pass

    return jsonify({
        "total_coins": len(SYMBOLS),
        "bullish":     bullish,
        "bearish":     bearish,
        "avg_change":  avg,
        "market_mood": "BULLISH" if avg > 0.5 else "BEARISH" if avg < -0.5 else "NEUTRAL",
        "fear_greed":  fg_score,
        "fg_label":    fg_label,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/log")
def api_log():
    try:
        if Path(LOG_FILE).exists():
            lines = Path(LOG_FILE).read_text().splitlines()
            return jsonify({"lines": lines[-200:]})
    except Exception:
        pass
    return jsonify({"lines": ["Log file not found — check GitHub Actions for logs"]})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    def run():
        try:
            import subprocess
            subprocess.run(["python", "trade_executor.py"], timeout=300)
        except Exception as e:
            log.error(f"Manual scan error: {e}")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "message": "Scan triggered"})


@app.route("/api/close_trade", methods=["POST"])
def api_close():
    data   = request.get_json()
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"ok": False, "error": "symbol required"}), 400

    trades = load_json(TRADES_FILE, {})
    if symbol not in trades:
        return jsonify({"ok": False, "error": f"{symbol} not found in open trades"}), 404

    trade = trades[symbol]

    try:
        from trade_executor import init_exchange
        ex   = init_exchange()

        for key, oid in trade.get("order_ids", {}).items():
            if key != "entry":
                try:
                    ex.cancel_order(oid, symbol)
                    log.info(f"  Cancelled {key} order {oid}")
                except Exception:
                    pass

        side = "sell" if trade["signal"] == "BUY" else "buy"
        close_order = ex.create_order(symbol, "market", side, trade["qty"])

        close_price = (
            float(close_order.get("average") or 0) or
            float(close_order.get("price")   or 0) or
            float(trade.get("entry", 0))
        )

        entry = float(trade["entry"])
        qty   = float(trade["qty"])
        pnl   = round(
            (close_price - entry) * qty if trade["signal"] == "BUY"
            else (entry - close_price) * qty,
            4
        )

        history = load_json(HISTORY_FILE, [])
        history.append({
            **trade,
            "close_price":  close_price,
            "pnl":          pnl,
            "closed_at":    datetime.now(timezone.utc).isoformat(),
            "close_reason": "Manual close",
        })
        save_json(HISTORY_FILE, history)

        trades.pop(symbol, None)
        save_json(TRADES_FILE, trades)

        return jsonify({
            "ok":          True,
            "pnl":         pnl,
            "close_price": close_price,
            "message":     f"Trade closed at ${close_price:.4f} | PnL: {pnl:+.4f} USDT",
        })

    except Exception as e:
        log.error(f"Close trade error for {symbol}: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/persistence/restore", methods=["POST"])
def api_persistence_restore():
    try:
        from persistence import pull_all_from_github
        count = pull_all_from_github()
        return jsonify({"ok": True, "restored": count})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/")
def root():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("dashboard_static", path)


# ── STARTUP TASKS (Runs automatically when Gunicorn boots) ──
try:
    restore_on_startup()
    threading.Thread(target=telegram_listener, daemon=True).start()
    threading.Thread(target=automated_scanner, daemon=True).start() # 👈 ADDED: The 15 min heart-beat
    log.info("✅ GitHub Restore, Telegram, and Auto-Scanner started successfully!")
except Exception as e:
    log.error(f"Startup task error: {e}")

# (This block is only used if you run it locally on your laptop)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard API → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
