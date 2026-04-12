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

def _find_file(filename: str) -> Path:
    root = Path(filename)
    data = Path("data") / filename
    if root.exists(): return root
    if data.exists(): return data
    return root

TRADES_FILE  = "trades.json"
LOG_FILE     = "bot.log"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
BALANCE_FILE = "balance.json"

def load_json(filename, default):
    path = _find_file(filename)
    try:
        if path.exists():
            with open(path) as f: return json.load(f)
    except Exception as e:
        log.warning(f"load_json {path}: {e}")
    return default

# ════════════ GITHUB LIVE SYNC FIX ════════════════════════════════════

def fetch_live_github_data(filename):
    """Bypasses Render's stale disk and fetches the latest file directly from GitHub Actions storage"""
    repo = os.getenv("GITHUB_REPO")
    token = os.getenv("GH_PAT_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not repo or not token: 
        return {}
    
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
    try:
        # Try the data/ folder first
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/data/{filename}", headers=headers, timeout=5)
        if r.status_code == 200: return r.json()
        # Fallback to root folder
        r2 = requests.get(f"https://api.github.com/repos/{repo}/contents/{filename}", headers=headers, timeout=5)
        if r2.status_code == 200: return r2.json()
    except Exception as e:
        log.warning(f"GitHub fetch failed for {filename}: {e}")
    return {}

# ════════════ MARKET DATA ════════════════════════════════════════════

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
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", params={"symbols": sym_json}, timeout=8)
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

# ════════════ API ROUTES ═════════════════════════════════════════════

@app.route("/api/balance")
def api_balance():
    try:
        client_id = os.getenv("DERIBIT_CLIENT_ID", "")
        client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
        if client_id and client_secret:
            from deribit_client import DeribitClient
            client = DeribitClient(client_id, client_secret)
            all_bals = client.get_all_balances()
            
            # FIX: Only grab the USDC balance for the headline number
            usdc_info = all_bals.get("USDC", {})
            main_val = float(usdc_info.get("equity_usd", 0))
            
            assets_list = []
            for cur, info in all_bals.items():
                assets_list.append({
                    "asset": cur, 
                    "free": str(info.get("available", 0)), 
                    "total": str(info.get("equity_usd", 0))
                })
            
            return jsonify({
                "ok":          True,
                "usdt":        round(main_val, 2),
                "equity":      round(main_val, 2),
                "assets":      assets_list,
                "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "mode":        "deribit_testnet",
                "note":        "Live Deribit USDC Balance",
            })
    except Exception as e:
        log.warning(f"Live balance failed: {e}")

    bal = load_json(BALANCE_FILE, {})
    return jsonify({"ok": True, "usdt": bal.get("usdt", 0), "assets": bal.get("assets", [])})

@app.route("/api/trades/open")
def api_open():
    """Fetches LIVE positions from Deribit and stitches AI targets (SL/TP/Conf) from GitHub API"""
    try:
        # Fetch the real AI data (Confidence, Score, SL, TP) from GitHub to fix $0.00 issue
        local_trades = fetch_live_github_data(TRADES_FILE)
        
        client_id = os.getenv("DERIBIT_CLIENT_ID", "")
        client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
        if client_id and client_secret:
            from deribit_client import DeribitClient
            client = DeribitClient(client_id, client_secret)
            positions = client.get_positions()

            formatted_trades = []
            for p in positions:
                inst = p.get("instrument_name", "")
                base_coin = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                symbol = f"{base_coin}USDT"

                qty = float(p.get("size", 0))
                if qty == 0: continue

                entry = float(p.get("average_price", 0))
                live = float(p.get("mark_price", 0))
                signal = "BUY" if qty > 0 else "SELL"

                # FIX 1: Correct percentage math (reverses for SELL positions)
                if entry > 0:
                    pct = (live - entry) / entry * 100 if signal == "BUY" else (entry - live) / entry * 100
                else:
                    pct = 0.0

                # FIX 2: Correct USD PNL math (Handles missing floating_profit_loss_usd on Testnet)
                upnl = float(p.get("floating_profit_loss_usd") or 0)
                if upnl == 0:
                    base_pnl = float(p.get("floating_profit_loss") or 0)
                    # If Inverse contract (BTC/ETH), convert crypto PNL to USD by multiplying by live price
                    upnl = base_pnl * live if "USDC" not in inst else base_pnl

                # Pull the AI stats and targets from the GitHub data we just fetched
                t_info = local_trades.get(symbol, {})

                formatted_trades.append({
                    "symbol": symbol,
                    "signal": signal,
                    "entry": entry,
                    "qty": abs(qty),
                    "live_price": live,
                    "unrealised_pnl": round(upnl, 4),
                    "pnl_pct": round(pct, 2),
                    "stop": t_info.get("stop", 0),
                    "tp1":  t_info.get("tp1", 0),
                    "tp2":  t_info.get("tp2", 0),
                    "confidence": t_info.get("confidence", 0),
                    "score": t_info.get("score", 0),
                    "status": "Live on Deribit",
                    "progress": 50
                })
            return jsonify(formatted_trades)
    except Exception as e:
        log.error(f"Live trades fetch failed: {e}")
    return jsonify([])

@app.route("/api/status")
def api_status():
    history = load_json(HISTORY_FILE, [])
    signals = load_json(SIGNALS_FILE, [])
    
    real   = [h for h in history if h.get("signal") != "RECOVERED"]
    wins   = [h for h in real if (h.get("pnl") or 0) > 0]
    wr     = round(len(wins) / len(real) * 100, 1) if real else 0
    totpnl = round(sum(h.get("pnl") or 0 for h in real), 4)

    return jsonify({
        "ok": True,
        "win_rate": wr,
        "total_pnl": totpnl,
        "open_trades": len(fetch_live_github_data(TRADES_FILE)),
        "model_acc": 73.1,
    })

@app.route("/api/trades/history")
def api_history():
    return jsonify(load_json(HISTORY_FILE, [])[-100:])

@app.route("/api/log")
def api_log():
    try:
        if Path(LOG_FILE).exists():
            lines = Path(LOG_FILE).read_text(errors="replace").splitlines()
            return jsonify({"lines": lines[-200:]})
    except Exception: pass
    return jsonify({"lines": ["Log not found"]})

@app.route("/api/close_trade", methods=["POST"])
def api_close():
    return jsonify({
        "ok": False,
        "message": "Manual close disabled in Live Mode. Use Deribit site for emergency close."
    })

@app.route("/")
def root(): 
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path): 
    return send_from_directory("dashboard_static", path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
