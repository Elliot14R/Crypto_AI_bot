# dashboard_api.py — Complete fixed version
# Fix 1: GH_PAT_TOKEN (matches Render env var)
# Fix 2: balance read from balance.json written by GitHub Actions
# Fix 3: No exchange calls from Render (India geo-block bypass)
# Fix 4: Scan button triggers GitHub Actions correctly

import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
log = logging.getLogger(__name__)

TRADES_FILE  = "trades.json"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
BALANCE_FILE = "balance.json"
LOG_FILE     = "bot.log"


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


# ── Price cache (public Binance API — no auth, no geo-block) ─
_price_cache = {}
_cache_time  = 0

def get_live_prices(symbols):
    global _price_cache, _cache_time
    if time.time() - _cache_time < 10 and _price_cache:
        return {s: _price_cache.get(s) for s in symbols}
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                         params={"symbols": json.dumps(list(symbols))}, timeout=8)
        if r.ok:
            for item in r.json():
                _price_cache[item["symbol"]] = {
                    "price":      float(item["lastPrice"]),
                    "change_pct": float(item["priceChangePercent"]),
                    "high":       float(item["highPrice"]),
                    "low":        float(item["lowPrice"]),
                    "volume":     float(item["quoteVolume"]),
                }
            _cache_time = time.time()
    except Exception as e:
        log.warning(f"Price fetch: {e}")
    return {s: _price_cache.get(s) for s in symbols}


def enrich_trades(trades, prices):
    result = []
    for sym, t in trades.items():
        entry  = t.get("entry", 0)
        qty    = t.get("qty",   0)
        signal = t.get("signal", "BUY")
        live   = (prices.get(sym) or {}).get("price")

        if live and entry and qty:
            upnl = round((live-entry)*qty if signal=="BUY" else (entry-live)*qty, 4)
            pct  = round((live-entry)/entry*100 if signal=="BUY" else (entry-live)/entry*100, 2)
        else:
            upnl = pct = 0

        tp2  = t.get("tp2", 0); sl = t.get("stop", 0)
        if signal=="BUY" and tp2>entry>sl and live:
            prog = round(max(0, min(100, (live-entry)/(tp2-entry)*100)), 1)
        elif signal=="SELL" and tp2<entry<sl and live:
            prog = round(max(0, min(100, (entry-live)/(entry-tp2)*100)), 1)
        else:
            prog = 0

        result.append({**t, "live_price":live, "unrealised_pnl":upnl,
                        "pnl_pct":pct, "progress":prog,
                        "status":"TP1 hit" if t.get("tp1_hit") else "Open"})
    return result


def trigger_github_scan():
    """
    Uses GH_PAT_TOKEN — the name in Render environment variables.
    Falls back to GITHUB_TOKEN as secondary option.
    """
    token = os.getenv("GH_PAT_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
    repo  = os.getenv("GITHUB_REPO", "")
    branch = os.getenv("GITHUB_BRANCH", "main")

    if not token:
        return False, ("GH_PAT_TOKEN not set in Render environment. "
                       "Add it at: Render dashboard → Your service → Environment")
    if not repo:
        return False, ("GITHUB_REPO not set in Render environment. "
                       "Set it to: Elliot14R/Crypto_AI_bot")

    workflow = os.getenv("GITHUB_WORKFLOW_FILE", "crypto_bot.yml")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    try:
        r = requests.post(url,
            headers={"Authorization": f"token {token}",
                     "Accept": "application/vnd.github.v3+json",
                     "X-GitHub-Api-Version": "2022-11-28"},
            json={"ref": branch}, timeout=10)
        if r.status_code == 204:
            return True, "✅ Scan triggered on GitHub Actions!"
        elif r.status_code == 401:
            return False, "GH_PAT_TOKEN is invalid or expired. Generate a new one at github.com → Settings → Developer Settings → Personal Access Tokens (repo scope)"
        elif r.status_code == 404:
            return False, f"Workflow '{workflow}' not found. Check .github/workflows/ folder in your repo"
        else:
            return False, f"GitHub API error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Request failed: {e}"


def telegram_listener():
    token = os.getenv("TELEGRAM_TOKEN", "")
    if not token: return
    offset = None
    while True:
        try:
            params = {"timeout":10, "allowed_updates":["message"]}
            if offset: params["offset"] = offset
            r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                             params=params, timeout=15)
            if r.ok:
                for item in r.json().get("result", []):
                    offset  = item["update_id"] + 1
                    msg     = item.get("message", {})
                    text    = msg.get("text", "").strip()
                    chat_id = msg.get("chat", {}).get("id")
                    if not chat_id: continue

                    def reply(txt):
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                            data={"chat_id":chat_id,"text":txt,"parse_mode":"Markdown"}, timeout=10)

                    if text == "/status":
                        trades = load_json(TRADES_FILE, {})
                        bal    = load_json(BALANCE_FILE, {})
                        txt    = f"📡 *Bot Status*\n💰 Balance: `{bal.get('usdt','?')} USDT`\n📂 Open trades: {len(trades)}"
                        for sym, t in trades.items():
                            txt += f"\n• {sym} ({t.get('signal','?')}) @ {t.get('entry','?')}"
                        reply(txt)
                    elif text == "/scan":
                        ok, m = trigger_github_scan()
                        reply(("✅ " if ok else "❌ ") + m)
                    elif text == "/balance":
                        bal = load_json(BALANCE_FILE, {})
                        reply(f"💰 Balance: *{bal.get('usdt','not fetched')} USDT*\nUpdated: {bal.get('updated_at','unknown')}")
        except Exception: pass
        time.sleep(3)


# ══════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════

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
    wr     = round(len(wins)/total*100, 1) if total else 0
    tpnl   = round(sum(h.get("pnl") or 0 for h in real), 4)

    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_sig = [s for s in signals
                 if (s.get("generated_at",""))[:10] == today and not s.get("rejected")]

    try:
        with open("model_performance.json") as f:
            model_acc = round(json.load(f).get("test_accuracy", 0.731)*100, 1)
    except Exception:
        model_acc = 73.1

    mode_data = load_json("scan_mode.json", {})

    return jsonify({
        "ok": True,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "open_trades":   len(trades),
        "max_trades":    3,
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      wr,
        "total_pnl":     tpnl,
        "model_acc":     model_acc,
        "today_signals": len(today_sig),
        "today_buy":     len([s for s in today_sig if s.get("signal")=="BUY"]),
        "today_sell":    len([s for s in today_sig if s.get("signal")=="SELL"]),
        "scan_mode":     mode_data.get("mode", "active"),
        "balance_usdt":  bal.get("usdt"),
        "balance_updated": bal.get("updated_at"),
    })


@app.route("/api/balance")
def api_balance():
    """
    Reads balance.json — written by trade_executor.py on GitHub Actions (US IP).
    Never calls Binance directly (avoids India geo-block on Render).
    """
    bal = load_json(BALANCE_FILE, {})
    if not bal or bal.get("usdt") is None:
        return jsonify({
            "ok":    False,
            "usdt":  None,
            "equity": None,
            "unrealised": 0,
            "assets": [],
            "note":  "Go to GitHub → Actions → Run workflow to trigger first scan. Balance will appear after.",
        })

    trades  = load_json(TRADES_FILE, {})
    prices  = get_live_prices(list(trades.keys())) if trades else {}
    upnl    = 0.0
    for sym, t in trades.items():
        live = (prices.get(sym) or {}).get("price")
        if live and t.get("entry") and t.get("qty"):
            upnl += (live-t["entry"])*t["qty"] if t["signal"]=="BUY" \
                    else (t["entry"]-live)*t["qty"]

    usdt = float(bal.get("usdt", 0))
    return jsonify({
        "ok":         True,
        "usdt":       usdt,
        "equity":     round(usdt + upnl, 2),
        "unrealised": round(upnl, 4),
        "assets":     bal.get("assets", []),
        "updated_at": bal.get("updated_at"),
    })


@app.route("/api/trades/open")
def api_open():
    trades  = load_json(TRADES_FILE, {})
    prices  = get_live_prices(list(trades.keys())) if trades else {}
    return jsonify(enrich_trades(trades, prices))


@app.route("/api/trades/history")
def api_history():
    return jsonify(load_json(HISTORY_FILE, [])[-100:])


@app.route("/api/signals")
def api_signals():
    signals  = load_json(SIGNALS_FILE, [])
    if request.args.get("symbol"):
        signals = [s for s in signals if s.get("symbol")==request.args["symbol"]]
    if request.args.get("type"):
        signals = [s for s in signals if s.get("signal")==request.args["type"].upper()]
    if request.args.get("rejected")=="false":
        signals = [s for s in signals if not s.get("rejected")]
    return jsonify(list(reversed(signals[-200:])))


@app.route("/api/log")
def api_log():
    try:
        if Path(LOG_FILE).exists():
            lines = Path(LOG_FILE).read_text(errors="replace").splitlines()
            return jsonify({"lines": lines[-200:]})
    except Exception: pass
    return jsonify({"lines": ["Log file not found — check GitHub Actions artifacts for bot.log"]})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    ok, message = trigger_github_scan()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/close_trade", methods=["POST"])
def api_close():
    """Cannot close from Render — Binance blocks India IPs."""
    data   = request.get_json() or {}
    symbol = data.get("symbol", "?")
    return jsonify({
        "ok":     False,
        "message": f"Cannot close {symbol} from Render server — Binance geo-blocks India IPs.",
        "manual":  "Trades close automatically via SL/TP. To close manually: testnet.binance.vision → Open Orders.",
    })


@app.route("/")
def root():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("dashboard_static", path)


# ══════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════

log.info("=" * 50)
log.info("CryptoBot Dashboard starting...")
log.info(f"GH_PAT_TOKEN:  {'SET ✅' if os.getenv('GH_PAT_TOKEN') else 'MISSING ❌'}")
log.info(f"GITHUB_REPO:   {os.getenv('GITHUB_REPO', 'NOT SET ❌')}")
log.info(f"TELEGRAM:      {'SET ✅' if os.getenv('TELEGRAM_TOKEN') else 'NOT SET'}")
log.info(f"balance.json:  {'EXISTS ✅' if Path(BALANCE_FILE).exists() else 'not yet (run a scan)'}")
log.info("=" * 50)

threading.Thread(target=telegram_listener, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
