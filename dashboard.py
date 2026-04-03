# dashboard_api.py — Final Fixed Version
# Fixes:
#   1. Reads balance.json from BOTH root AND data/ subfolder (handles both locations)
#   2. Handles paper trading balance format (usdt as int/float, no error field)
#   3. Better startup logging so you can see exactly what's happening
#   4. GH_PAT_TOKEN support for scan button
#   5. All timeouts 8s (internal reads are instant — only external calls need timeouts)

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
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── File paths — check both root AND data/ subfolder ─────────────────
# GitHub Actions cache saves to root; some repo setups use data/ folder
def _find_file(filename: str) -> Path:
    """Find a state file — checks root first, then data/ subfolder."""
    root = Path(filename)
    data = Path("data") / filename
    if root.exists():
        return root
    if data.exists():
        return data
    return root   # default to root (will be created there)

TRADES_FILE  = "trades.json"
LOG_FILE     = "bot.log"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
BALANCE_FILE = "balance.json"


# ════════════ FILE I/O ═══════════════════════════════════════════════

def load_json(filename, default):
    """Load JSON — checks root then data/ subfolder."""
    path = _find_file(filename)
    try:
        if path.exists():
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


# ════════════ PRICE FETCHING (public API — no geo-block) ═════════════

_price_cache   = {}
_price_cache_t = 0
PRICE_TTL      = 10

def get_live_prices(symbols):
    global _price_cache, _price_cache_t
    now = time.time()
    if now - _price_cache_t < PRICE_TTL and _price_cache:
        return {s: _price_cache.get(s) for s in symbols}
    try:
        sym_json = json.dumps(list(symbols))
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                         params={"symbols": sym_json}, timeout=8)
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
        log.warning(f"Price fetch: {e}")
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
    # Try both token env var names
    token = (os.getenv("GH_PAT_TOKEN", "") or
             os.getenv("GITHUB_TOKEN",  "") or
             os.getenv("GH_TOKEN",      ""))
    repo  = os.getenv("GITHUB_REPO", "")
    workflow = os.getenv("GITHUB_WORKFLOW", "crypto_bot.yml")

    if not token:
        return False, "GH_PAT_TOKEN not set in Render environment variables"
    if not repo:
        return False, "GITHUB_REPO not set in Render environment variables"

    try:
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
        r   = requests.post(url,
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github.v3+json"},
            json={"ref": os.getenv("GITHUB_BRANCH", "main")},
            timeout=10)
        if r.status_code == 204:
            return True, "✅ Scan triggered on GitHub Actions!"
        elif r.status_code == 401:
            return False, "GH_PAT_TOKEN is invalid or expired"
        elif r.status_code == 404:
            return False, f"Workflow '{workflow}' not found in repo"
        else:
            return False, f"GitHub API {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return False, str(e)


# ════════════ TELEGRAM ═══════════════════════════════════════════════

def telegram_listener():
    token = os.getenv("TELEGRAM_TOKEN", "")
    if not token:
        return
    offset = None
    while True:
        try:
            r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                params={"timeout": 10, "allowed_updates": ["message"],
                        **({"offset": offset} if offset else {})},
                timeout=15)
            if r.ok:
                for item in r.json().get("result", []):
                    offset  = item["update_id"] + 1
                    msg     = item.get("message", {})
                    text    = msg.get("text", "").strip()
                    chat_id = msg.get("chat", {}).get("id")
                    if not chat_id: continue

                    if text == "/status":
                        trades = load_json(TRADES_FILE, {})
                        bal    = load_json(BALANCE_FILE, {})
                        usdt   = bal.get("usdt", "?")
                        mode   = bal.get("mode", "testnet")
                        reply  = f"📡 *Bot Status*\nBalance: `{usdt}` USDT ({mode})\nOpen: {len(trades)} trade(s)\n"
                        for sym, t in trades.items():
                            reply += f"• {sym} {t.get('signal','?')} @ {t.get('entry','?')}\n"
                        if not trades:
                            reply += "_No open trades_"
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id, "text": reply, "parse_mode": "Markdown"},
                            timeout=10)

                    elif text == "/scan":
                        ok, msg_txt = trigger_github_scan()
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id, "text": msg_txt}, timeout=10)

                    elif text == "/balance":
                        bal     = load_json(BALANCE_FILE, {})
                        usdt    = bal.get("usdt", "Not available")
                        equity  = bal.get("equity", usdt)
                        updated = bal.get("updated_at", "unknown")
                        mode    = bal.get("mode", "testnet")
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id": chat_id,
                                  "text": f"💰 *Balance: {usdt} USDT*\nEquity: {equity} USDT\nMode: {mode}\nUpdated: {updated}",
                                  "parse_mode": "Markdown"},
                            timeout=10)
        except Exception:
            pass
        time.sleep(3)


# ════════════ API ROUTES ═════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    trades  = load_json(TRADES_FILE,  {})
    history = load_json(HISTORY_FILE, [])
    signals = load_json(SIGNALS_FILE, [])
    bal     = load_json(BALANCE_FILE, {})

    real   = [h for h in history if h.get("signal") != "RECOVERED"]
    wins   = [h for h in real if (h.get("pnl") or 0) > 0]
    losses = [h for h in real if (h.get("pnl") or 0) <= 0]
    total  = len(real)
    wr     = round(len(wins) / total * 100, 1) if total else 0
    totpnl = round(sum(h.get("pnl") or 0 for h in real), 4)

    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_sigs = [s for s in signals
                  if (s.get("generated_at") or "")[:10] == today
                  and not s.get("rejected")]

    try:
        with open("model_performance.json") as f:
            model_acc = round(json.load(f).get("test_accuracy", 0.731) * 100, 1)
    except Exception:
        model_acc = 73.1

    mode_data = load_json("scan_mode.json", {})

    return jsonify({
        "ok":              True,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "open_trades":     len(trades),
        "max_trades":      3,
        "total_closed":    total,
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        wr,
        "total_pnl":       totpnl,
        "model_acc":       model_acc,
        "today_signals":   len(today_sigs),
        "today_buy":       len([s for s in today_sigs if s.get("signal") == "BUY"]),
        "today_sell":      len([s for s in today_sigs if s.get("signal") == "SELL"]),
        "scan_mode":       mode_data.get("mode", "active"),
        "balance_usdt":    bal.get("usdt"),
        "balance_updated": bal.get("updated_at"),
        "trading_mode":    bal.get("mode", "testnet"),
    })


@app.route("/api/balance")
def api_balance():
    """
    Reads balance.json written by GitHub Actions paper_trader / trade_executor.
    Checks both root and data/ subfolder.
    Handles paper trading format AND testnet format.
    """
    bal = load_json(BALANCE_FILE, {})
    bal_path = str(_find_file(BALANCE_FILE))

    log.info(f"Balance request — path: {bal_path} — data: {bal}")

    # File truly missing
    if not bal:
        return jsonify({
            "ok":         False,
            "usdt":       None,
            "equity":     None,
            "unrealised": 0,
            "assets":     [],
            "note":       "balance.json not found. Run a GitHub Actions scan first.",
            "file_path":  bal_path,
        })

    # Has error from failed exchange call
    if bal.get("error"):
        return jsonify({
            "ok":         False,
            "usdt":       None,
            "equity":     None,
            "unrealised": 0,
            "assets":     [],
            "error":      bal.get("error"),
            "note":       "Balance fetch failed. Check GitHub Actions log.",
            "updated_at": bal.get("updated_at"),
        })

    # ── Success — handle both paper trading and testnet formats ──────
    usdt       = float(bal.get("usdt") or 0)
    equity     = float(bal.get("equity")     or usdt)
    unrealised = float(bal.get("unrealised") or 0)
    assets     = bal.get("assets", [])
    mode       = bal.get("mode", "testnet")

    # If no assets array but we have usdt, create a minimal one
    if not assets and usdt > 0:
        assets = [{"asset": "USDT", "free": str(round(usdt, 2)),
                   "total": str(round(equity, 2))}]

    return jsonify({
        "ok":          True,
        "usdt":        round(usdt, 2),
        "equity":      round(equity, 2),
        "unrealised":  round(unrealised, 4),
        "assets":      assets,
        "updated_at":  bal.get("updated_at"),
        "mode":        mode,
        "note":        f"Paper trading — real signals, simulated fills" if mode == "paper_trading"
                       else "Testnet balance",
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
        "bearish":     len(SYMBOLS) - bullish,
        "avg_change":  avg,
        "market_mood": "BULLISH" if avg > 0.5 else "BEARISH" if avg < -0.5 else "NEUTRAL",
        "fear_greed":  fg_score,
        "fg_label":    fg_label,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/log")
def api_log():
    try:
        # Check both root and logs/ subfolder
        for log_path in ["bot.log", "data/bot.log", "logs/bot.log"]:
            if Path(log_path).exists():
                lines = Path(log_path).read_text(errors="replace").splitlines()
                return jsonify({"lines": lines[-200:]})
    except Exception:
        pass
    return jsonify({"lines": ["Log not found — check GitHub Actions artifacts for bot.log"]})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    ok, message = trigger_github_scan()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/close_trade", methods=["POST"])
def api_close():
    data   = request.get_json() or {}
    symbol = data.get("symbol", "?")
    mode   = load_json(BALANCE_FILE, {}).get("mode", "testnet")

    if mode == "paper_trading":
        # Paper trading: can close directly by updating trades.json
        trades = load_json(TRADES_FILE, {})
        if symbol in trades:
            trade = trades[symbol]
            from paper_trader import get_live_price, PaperTrader
            live  = get_live_price(symbol)
            entry = float(trade.get("entry", live))
            qty   = float(trade.get("qty", 0))
            pnl   = round((live - entry) * qty if trade.get("signal") == "BUY"
                          else (entry - live) * qty, 4)
            pt = PaperTrader()
            pt.update_balance_after_close(pnl)
            history = load_json(HISTORY_FILE, [])
            history.append({**trade, "close_price": live, "pnl": pnl,
                             "closed_at": datetime.now(timezone.utc).isoformat(),
                             "close_reason": "Manual close (dashboard)"})
            save_json(HISTORY_FILE, history)
            trades.pop(symbol, None)
            save_json(TRADES_FILE, trades)
            return jsonify({"ok": True, "pnl": pnl, "close_price": live,
                            "message": f"Paper trade closed @ {live:.4f} | PnL: {pnl:+.4f} USDT"})
        return jsonify({"ok": False, "error": f"{symbol} not in open trades"}), 404

    return jsonify({
        "ok":     False,
        "message": f"Cannot close {symbol} from dashboard — geo-restricted.",
        "action":  "Trades close automatically when SL/TP is hit. Or cancel at testnet.binance.vision.",
    }), 200


@app.route("/api/debug/balance")
def api_debug_balance():
    """Debug endpoint — shows exactly where balance.json is being read from."""
    root_path = Path(BALANCE_FILE)
    data_path = Path("data") / BALANCE_FILE
    return jsonify({
        "root_exists":  root_path.exists(),
        "data_exists":  data_path.exists(),
        "reading_from": str(_find_file(BALANCE_FILE)),
        "content":      load_json(BALANCE_FILE, None),
        "cwd":          str(Path.cwd()),
    })


@app.route("/")
def root():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("dashboard_static", path)


# ════════════ STARTUP DIAGNOSTICS ════════════════════════════════════

def _startup_check():
    """Run at boot — log exactly what files are found and their content."""
    log.info("=" * 50)
    log.info("CryptoBot Dashboard API starting...")

    files_to_check = [
        ("balance.json",      BALANCE_FILE),
        ("trades.json",       TRADES_FILE),
        ("signals.json",      SIGNALS_FILE),
        ("trade_history.json",HISTORY_FILE),
    ]
    for label, fname in files_to_check:
        path = _find_file(fname)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if fname == BALANCE_FILE:
                    usdt = data.get("usdt", "?")
                    mode = data.get("mode", "testnet")
                    log.info(f"  ✅ {label}: EXISTS — {usdt} USDT ({mode}) @ {path}")
                else:
                    size = len(data) if isinstance(data, (list, dict)) else "?"
                    log.info(f"  ✅ {label}: EXISTS — {size} entries @ {path}")
            except Exception as e:
                log.warning(f"  ⚠️ {label}: EXISTS but unreadable — {e}")
        else:
            log.info(f"  ℹ️  {label}: not yet (run a scan to create it) — checked {path}")

    tok  = "SET ✅" if os.getenv("TELEGRAM_TOKEN")  else "MISSING ❌"
    repo = os.getenv("GITHUB_REPO", "NOT SET ❌")
    pat  = "SET ✅" if (os.getenv("GH_PAT_TOKEN") or os.getenv("GITHUB_TOKEN")) else "MISSING ❌"
    log.info(f"  TELEGRAM: {tok}")
    log.info(f"  GITHUB_REPO: {repo}")
    log.info(f"  GH_PAT_TOKEN: {pat}")
    log.info("=" * 50)

_startup_check()

try:
    threading.Thread(target=telegram_listener, daemon=True).start()
except Exception as e:
    log.error(f"Telegram listener failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
