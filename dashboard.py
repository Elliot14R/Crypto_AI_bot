# dashboard.py 
import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)

# ── THE FIX: Writes to bot.log AND the Render Console ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("bot.log", mode='a'), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

TRADES_FILE  = "trades.json"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
BALANCE_FILE = "balance.json"

# ════════════ FILE I/O ═══════════════════════════════════════
def load_json(path, default):
    try:
        from persistence import load_from_github
        data = load_from_github(path, None)
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

# ════════════ PRICE FETCHING ═════════════════════════════════
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
        live   = pd_.get("price")

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

# ════════════ GITHUB ACTIONS TRIGGER ════════════════════════
def trigger_github_scan():
    token = os.getenv("GH_PAT_TOKEN",  "") or os.getenv("GITHUB_TOKEN",  "")
    repo  = os.getenv("GITHUB_REPO", "Elliot14R/Crypto_AI_bot")

    if not token:
        return False, "GH_PAT_TOKEN not set in Render environment variables."

    try:
        workflow = os.getenv("GITHUB_WORKFLOW", "crypto_bot.yml")
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
        r   = requests.post(
            url,
            headers={
                "Authorization": f"token {token}",
                "Accept":        "application/vnd.github.v3+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"ref": os.getenv("GITHUB_BRANCH", "main")},
            timeout=10,
        )
        if r.status_code == 204:
            log.info(f"✅ GitHub Actions scan triggered successfully")
            return True, "✅ Scan triggered on GitHub Actions! Check Actions tab."
        else:
            return False, f"GitHub API error {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"

# ════════════ API ROUTES ═════════════════════════════════════
@app.route("/api/status")
def api_status():
    trades  = load_json(TRADES_FILE,  {})
    history = load_json(HISTORY_FILE, [])
    signals = load_json(SIGNALS_FILE, [])

    real   = [h for h in history if h.get("signal") != "RECOVERED"]
    wins   = [h for h in real if (h.get("pnl") or 0) > 0]
    losses = [h for h in real if (h.get("pnl") or 0) <= 0]
    total  = len(real)
    wr     = round(len(wins) / total * 100, 1) if total else 0
    totpnl = round(sum(h.get("pnl") or 0 for h in real), 4)

    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_sigs = [s for s in signals if (s.get("generated_at") or "")[:10] == today and not s.get("rejected")]

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
        "today_signals":   len(today_sigs),
        "scan_mode":       mode_data.get("mode", "active")
    })

@app.route("/api/balance")
def api_balance():
    bal = load_json(BALANCE_FILE, {})
    trades  = load_json(TRADES_FILE, {})
    symbols = list(trades.keys())
    prices  = get_live_prices(symbols) if symbols else {}
    upnl    = 0.0
    
    for sym, t in trades.items():
        live = (prices.get(sym) or {}).get("price")
        if live and t.get("entry") and t.get("qty"):
            upnl += (live - t["entry"]) * t["qty"] if t["signal"] == "BUY" else (t["entry"] - live) * t["qty"]

    usdt = float(bal.get("usdt", 0))
    return jsonify({
        "ok":         True,
        "usdt":       usdt,
        "equity":     round(usdt + upnl, 2),
        "unrealised": round(upnl, 4),
        "assets":     bal.get("assets", []),
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
    return jsonify(list(reversed(load_json(SIGNALS_FILE, [])[-200:])))

@app.route("/api/log")
def api_log():
    try:
        if Path("bot.log").exists():
            lines = Path("bot.log").read_text(errors="replace").splitlines()
            return jsonify({"lines": lines[-200:]})
    except Exception:
        pass
    return jsonify({"lines": ["Log file empty or not found yet. Run a scan!"]})

@app.route("/api/scan", methods=["POST"])
def api_scan():
    ok, message = trigger_github_scan()
    return jsonify({"ok": ok, "message": message}), 200

@app.route("/")
def root():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("dashboard_static", path)

# ════════════ STARTUP ════════════════════════════════════════
if __name__ == "__main__":
    # Pull the latest data from GitHub before starting the server
    try:
        from persistence import pull_all_from_github
        pull_all_from_github()
    except Exception as e:
        log.warning(f"Startup pull failed: {e}")
        
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard API → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
