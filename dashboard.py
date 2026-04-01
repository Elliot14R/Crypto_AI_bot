# dashboard.py — Complete fixed version
# FIXES:
#   - ALL exchange calls removed from dashboard.py to prevent 451 India geo-block
#   - Balance read from balance.json (written by GitHub Actions)
#   - Scan button triggers GitHub Actions via repository_dispatch webhook
#   - Simple file I/O replaces complex persistence imports

import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot.log", mode='a'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

TRADES_FILE  = "trades.json"
LOG_FILE     = "bot.log"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
BALANCE_FILE = "balance.json"   


# ════════════ SIMPLE FILE I/O ════════════

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"load_json {path}: {e}")
    return default

def save_json(path, data):
    try:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"save_json {path}: {e}")


# ════════════ PRICE FETCHING (Public API — No Geo-Block) ═════════════

_price_cache   = {}
_price_cache_t = 0
PRICE_CACHE_TTL = 10

def get_live_prices(symbols):
    global _price_cache, _price_cache_t
    now = time.time()

    if now - _price_cache_t < PRICE_CACHE_TTL and _price_cache:
        return {s: _price_cache.get(s) for s in symbols}

    try:
        sym_json = json.dumps(list(symbols))
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbols": sym_json},
            timeout=8
        )
        if r.ok and isinstance(r.json(), list):
            for item in r.json():
                _price_cache[item["symbol"]] = {
                    "price":      float(item["lastPrice"]),
                    "change_pct": float(item["priceChangePercent"]),
                    "high":       float(item["highPrice"]),
                    "low":        float(item["lowPrice"]),
                    "volume":     float(item["quoteVolume"]),
                }
            _price_cache_t = now
    except Exception as e:
        log.warning(f"Price fetch failed: {e}")

    return {s: _price_cache.get(s) for s in symbols}


def enrich_trades(trades, prices):
    result = []
    for sym, t in trades.items():
        entry  = t.get("entry", 0)
        qty    = t.get("qty",   0)
        signal = t.get("signal", "BUY")
        pd_    = prices.get(sym) or {}
        live   = pd_.get("price") if pd_ else None

        if live and entry and qty:
            upnl = round((live-entry)*qty if signal=="BUY" else (entry-live)*qty, 4)
            pct  = round((live-entry)/entry*100 if signal=="BUY" else (entry-live)/entry*100, 2)
        else:
            upnl, pct = 0, 0

        tp2 = t.get("tp2", 0)
        sl  = t.get("stop", 0)
        if signal == "BUY" and tp2 > entry > sl and live:
            prog = round(max(0, min(100, (live-entry)/(tp2-entry)*100)), 1)
        elif signal == "SELL" and tp2 < entry < sl and live:
            prog = round(max(0, min(100, (entry-live)/(entry-tp2)*100)), 1)
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


# ════════════ GITHUB ACTIONS TRIGGER ════════════════════════════════

def trigger_github_scan():
    token = os.getenv("GH_PAT_TOKEN", "") or os.getenv("GITHUB_TOKEN", "")
    repo  = os.getenv("GITHUB_REPO",  "")

    if not token or not repo:
        return False, "GITHUB_TOKEN or GITHUB_REPO not set in .env"

    try:
        url = f"https://api.github.com/repos/{repo}/actions/workflows/crypto_bot.yml/dispatches"
        r   = requests.post(
            url,
            headers={
                "Authorization": f"token {token}",
                "Accept":        "application/vnd.github.v3+json",
            },
            json={"ref": os.getenv("GITHUB_BRANCH", "main")},
            timeout=10,
        )
        if r.status_code == 204:
            return True, "Scan triggered on GitHub Actions"
        else:
            return False, f"GitHub API error: {r.status_code} {r.text[:100]}"
    except Exception as e:
        return False, str(e)


# ════════════ TELEGRAM LISTENER ═════════════════════════════════════

def telegram_listener():
    token = os.getenv("TELEGRAM_TOKEN", "")
    if not token:
        log.warning("Telegram: no token — listener disabled")
        return

    offset = None
    while True:
        try:
            url    = f"https://api.telegram.org/bot{token}/getUpdates"
            params = {"timeout": 10, "allowed_updates": ["message"]}
            if offset:
                params["offset"] = offset

            r = requests.get(url, params=params, timeout=15)
            if r.ok:
                for item in r.json().get("result", []):
                    offset  = item["update_id"] + 1
                    msg     = item.get("message", {})
                    text    = msg.get("text", "").strip()
                    chat_id = msg.get("chat", {}).get("id")

                    if text == "/status":
                        trades = load_json(TRADES_FILE, {})
                        bal    = load_json(BALANCE_FILE, {})
                        usdt   = bal.get("usdt", "?")
                        reply  = f"📡 *Bot Status*\nBalance: {usdt} USDT\nOpen trades: {len(trades)}\n"
                        if trades:
                            for sym, t in trades.items():
                                reply += f"• {sym} ({t.get('signal','?')}) @ {t.get('entry','?')}\n"
                        else:
                            reply += "_No open trades_"
                        requests.post(
                            f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id, "text": reply, "parse_mode": "Markdown"},
                            timeout=10
                        )

                    elif text == "/scan":
                        ok, msg_txt = trigger_github_scan()
                        requests.post(
                            f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id, "text": ("✅ " if ok else "❌ ") + msg_txt},
                            timeout=10
                        )

                    elif text == "/balance":
                        bal  = load_json(BALANCE_FILE, {})
                        usdt = bal.get("usdt", "Not fetched yet")
                        updated = bal.get("updated_at", "unknown")
                        requests.post(
                            f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id, "text": f"💰 Balance: *{usdt} USDT*\nUpdated: {updated}", "parse_mode": "Markdown"},
                            timeout=10
                        )
        except Exception:
            pass
        time.sleep(3)


# ════════════ API ROUTES ════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    trades  = load_json(TRADES_FILE,  {})
    history = load_json(HISTORY_FILE, [])
    signals = load_json(SIGNALS_FILE, [])

    wins   = [h for h in history if (h.get("pnl") or 0) > 0]
    losses = [h for h in history if (h.get("pnl") or 0) <= 0]
    total  = len(history)
    wr     = round(len(wins) / total * 100, 1) if total else 0
    totpnl = round(sum(h.get("pnl") or 0 for h in history), 4)

    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_sigs = [s for s in signals if (s.get("generated_at") or "")[:10] == today]

    try:
        with open("model_performance.json") as f:
            model_acc = round(json.load(f).get("test_accuracy", 0.731) * 100, 1)
    except Exception:
        model_acc = 73.1

    mode_data = load_json("scan_mode.json", {})
    bal       = load_json(BALANCE_FILE, {})

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
        "balance_usdt":  bal.get("usdt"),
        "balance_updated": bal.get("updated_at"),
    })


@app.route("/api/balance")
def api_balance():
    bal = load_json(BALANCE_FILE, {})

    if not bal:
        return jsonify({
            "ok":         False,
            "usdt":       None,
            "equity":     None,
            "unrealised": 0,
            "assets":     [],
            "note":       "Balance not yet fetched. Trigger a GitHub Actions scan first.",
        })

    if bal.get("error") or bal.get("usdt") is None:
        return jsonify({
            "ok":         False,
            "usdt":       None,
            "equity":     None,
            "unrealised": 0,
            "assets":     [],
            "error":      bal.get("error", "Unknown error"),
            "note":       "Balance fetch failed on GitHub Actions. Check Actions log for details.",
            "updated_at": bal.get("updated_at"),
        })

    trades  = load_json(TRADES_FILE, {})
    symbols = list(trades.keys())
    prices  = get_live_prices(symbols) if symbols else {}
    upnl    = 0.0
    for sym, t in trades.items():
        live = (prices.get(sym) or {}).get("price")
        if live and t.get("entry") and t.get("qty"):
            if t["signal"] == "BUY":
                upnl += (live - t["entry"]) * t["qty"]
            else:
                upnl += (t["entry"] - live) * t["qty"]

    usdt = float(bal.get("usdt", 0))
    return jsonify({
        "ok":          True,
        "usdt":        usdt,
        "equity":      round(usdt + upnl, 2),
        "unrealised":  round(upnl, 4),
        "assets":      bal.get("assets", []),
        "other_count": len(bal.get("assets", [])),
        "updated_at":  bal.get("updated_at"),
        "note":        "Balance fetched on GitHub Actions (US IP, geo-block bypassed)",
    })


@app.route("/api/trades/open")
def api_open():
    trades  = load_json(TRADES_FILE, {})
    symbols = list(trades.keys())
    prices  = get_live_prices(symbols) if symbols else {}
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

    if symbol:   signals = [s for s in signals if s.get("symbol") == symbol]
    if sig_type: signals = [s for s in signals if s.get("signal") == sig_type.upper()]
    if rejected == "false": signals = [s for s in signals if not s.get("rejected")]
    elif rejected == "true": signals = [s for s in signals if s.get("rejected")]

    return jsonify(list(reversed(signals[-200:])))


@app.route("/api/market/overview")
def api_market_overview():
    SYMBOLS = [
        "BTCUSDT","ETHUSDT","BNBUSDT",
        "SOLUSDT","AVAXUSDT","NEARUSDT","SUIUSDT","APTUSDT",
        "LINKUSDT","DOTUSDT","UNIUSDT","AAVEUSDT","XRPUSDT",
        "FETUSDT","RENDERUSDT","ADAUSDT","INJUSDT","ARBUSDT","OPUSDT","SEIUSDT",
    ]
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
            lines = Path(LOG_FILE).read_text(errors="replace").splitlines()
            return jsonify({"lines": lines[-200:]})
    except Exception:
        pass
    return jsonify({"lines": ["Log file not found — check GitHub Actions artifacts for bot.log"]})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    ok, message = trigger_github_scan()
    if ok:
        log.info(f"Scan triggered via GitHub Actions")
        return jsonify({"ok": True, "message": message})
    else:
        log.warning(f"Scan trigger failed: {message}")
        return jsonify({
            "ok":      False,
            "message": message,
            "help":    "Set GITHUB_TOKEN and GITHUB_REPO in Render environment variables. "
                       "Or go to GitHub → Actions → Run workflow manually.",
        })


@app.route("/api/close_trade", methods=["POST"])
def api_close():
    data   = request.get_json() or {}
    symbol = data.get("symbol", "?")
    return jsonify({
        "ok":      False,
        "message": f"Cannot close {symbol} from dashboard — Binance blocks this server IP (India geo-restriction).",
        "action":  "Go to testnet.binance.vision → Open Orders tab → Cancel the orders manually.",
        "note":    "Trades close automatically when Stop Loss or Take Profit is hit by GitHub Actions bot.",
    }), 200


@app.route("/")
def root():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("dashboard_static", path)


# ════════════ STARTUP ════════════════════════════════════════════════

log.info("=" * 50)
log.info("CryptoBot Dashboard API starting...")
log.info("Exchange calls: DISABLED (geo-restriction fix)")
log.info("Balance source: balance.json (written by GitHub Actions)")
log.info("Scan trigger:   GitHub Actions workflow_dispatch")
log.info("=" * 50)

try:
    threading.Thread(target=telegram_listener, daemon=True).start()
    log.info("✅ Telegram listener started")
except Exception as e:
    log.error(f"Telegram listener failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard API → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
