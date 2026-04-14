import os, json, base64, time, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

GH_TOKEN  = os.getenv("GH_PAT_TOKEN", "")
GH_REPO   = os.getenv("GITHUB_REPO",  "Elliot14R/Crypto_AI_bot")
GH_BRANCH = os.getenv("GITHUB_BRANCH", "main")

_cache, _cache_ts = {}, {}
CACHE_TTL = 60

# ── GitHub fetch (base64 decode, two-path fallback) ───────────────────

def gh_fetch(filename: str):
    """Fetch JSON/text from GitHub repo. Returns parsed object or None."""
    if not GH_TOKEN or not GH_REPO:
        return None
    headers = {"Authorization": f"token {GH_TOKEN}",
                "Accept": "application/vnd.github.v3+json"}
    for path in [f"data/{filename}", filename]:
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/contents/{path}",
                headers=headers, timeout=8
            )
            if r.status_code == 200:
                raw = base64.b64decode(r.json()["content"]).decode("utf-8")
                return json.loads(raw) if filename.endswith(".json") else raw
        except Exception as e:
            log.debug(f"gh_fetch {path}: {e}")
    return None

def get(filename: str, default):
    """Cache-backed getter: GitHub first, local disk fallback."""
    now = time.time()
    if filename in _cache and now - _cache_ts.get(filename, 0) < CACHE_TTL:
        return _cache[filename]
    data = gh_fetch(filename)
    if data is None:
        for p in [Path(filename), Path("data") / filename]:
            try:
                if p.exists():
                    txt = p.read_text()
                    data = json.loads(txt) if filename.endswith(".json") else txt
                    break
            except Exception:
                pass
    if data is not None:
        _cache[filename] = data
        _cache_ts[filename] = now
        return data
    return default

def bust(filename: str):
    _cache_ts[filename] = 0

# ── Deribit live fetch ─────────────────────────────────────────────────

def deribit_client():
    cid    = os.getenv("DERIBIT_CLIENT_ID", "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        return None
    try:
        from deribit_client import DeribitClient
        return DeribitClient(cid, secret)
    except Exception as e:
        log.warning(f"DeribitClient init: {e}")
        return None

# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index(): return send_from_directory("dashboard_static", "index.html")

@app.route("/trading")
@app.route("/signals")
@app.route("/market")
@app.route("/open-trades")
@app.route("/history")
@app.route("/performance")
@app.route("/configuration")
def spa(): 
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    try: return send_from_directory("dashboard_static", path)
    except: return send_from_directory("dashboard_static", "index.html")

# ── /api/status ────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    bust("trade_history.json"); bust("signals.json")
    bust("scan_mode.json");     bust("trades.json"); bust("balance.json")

    history   = get("trade_history.json", [])
    signals   = get("signals.json",       [])
    scan_mode = get("scan_mode.json",     {})
    trades    = get("trades.json",         {})
    balance   = get("balance.json",        {})

    real      = [h for h in history if h.get("signal") != "RECOVERED"]
    wins      = [h for h in real if (h.get("pnl") or 0) > 0]
    tpnl      = sum(h.get("pnl", 0) for h in real)
    win_rate = round(len(wins) / len(real) * 100, 1) if real else 0

    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs  = [s for s in signals if s.get("generated_at", "").startswith(today)]
    buys    = sum(1 for s in t_sigs if s.get("signal") == "BUY")
    sells   = sum(1 for s in t_sigs if s.get("signal") == "SELL")
    mode    = scan_mode.get("mode", "active")

    return jsonify({
        "win_rate": win_rate, "wins": len(wins),
        "losses": len(real) - len(wins), "total_pnl": round(tpnl, 4),
        "total_trades": len(real),
        "open_trades": len([t for t in trades.values() if not t.get("closed")]),
        "max_trades": 3, "scan_mode": mode, "mode_label": mode.upper(),
        "min_confidence": 65, "min_score": 2,
        "today_signals": len(t_sigs), "today_buys": buys, "today_sells": sells,
        "model_accuracy": 73.1,
        "balance": balance.get("usdt", 0),
        "exchange": balance.get("exchange", "Deribit Testnet"),
        "last_updated": balance.get("updated_at", ""),
    })

# ── /api/balance ────────────────────────────────────────────────────────
@app.route("/api/balance")
def api_balance():
    client = deribit_client()
    if client:
        try:
            bals  = client.get_all_balances()
            total = client.get_total_equity_usd()
            assets = [{"asset": c, "free": str(i.get("available", 0)),
                       "total": str(i.get("equity_usd", 0))}
                      for c, i in bals.items()]
            return jsonify({
                "ok": True, "usdt": round(total, 2), "equity": round(total, 2),
                "assets": assets,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "exchange": "Deribit(by Coinbase) Testnet",
            })
        except Exception as e:
            log.warning(f"Live balance: {e}")
    # File fallback
    bust("balance.json")
    bal = get("balance.json", {})
    return jsonify({**bal, "ok": True})

# ── /api/trades/open ────────────────────────────────────────────────────
@app.route("/api/trades/open")
def api_open_trades():
    """
    Fetches live positions from Deribit and stitches SL/TP/confidence
    from trades.json on GitHub. Returns full data for dashboard table.
    """
    bust("trades.json")
    ai_data = get("trades.json", {})

    # Live prices from Binance (for PnL enrichment)
    live_prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price", timeout=6)
        if r.ok:
            live_prices = {i["symbol"]: float(i["price"]) for i in r.json()}
    except Exception:
        pass

    client = deribit_client()
    if client:
        try:
            positions = client.get_positions()
            result = []
            for p in positions:
                size = float(p.get("size", 0))
                if size == 0:
                    continue
                inst      = p.get("instrument_name", "")
                base      = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                symbol    = f"{base}USDT"
                entry     = float(p.get("average_price", 0) or 0)
                live      = float(p.get("mark_price", 0) or 0)
                signal    = "BUY" if size > 0 else "SELL"

                # Unrealised PnL — prefer exchange value, fallback to calculation
                upnl = float(p.get("floating_profit_loss_usd") or
                             p.get("floating_profit_loss") or 0)

                pnl_pct = 0.0
                if entry > 0:
                    pnl_pct = ((live - entry) / entry * 100 if signal == "BUY"
                               else (entry - live) / entry * 100)

                # Stitch AI targets from GitHub trades.json
                t = ai_data.get(symbol, {})
                stop  = float(t.get("stop",  0) or 0)
                tp1   = float(t.get("tp1",   0) or 0)
                tp2   = float(t.get("tp2",   0) or 0)
                conf  = t.get("confidence", 0)
                score = t.get("score",      0)

                # Progress bar (0-100%) entry → TP2
                progress = 0.0
                if entry > 0 and tp2 > 0 and live > 0:
                    dist = abs(tp2 - entry)
                    if dist > 0:
                        progress = min(100, max(0, abs(live - entry) / dist * 100))

                result.append({
                    "symbol":     symbol,
                    "signal":     signal,
                    "entry":      round(entry, 6),
                    "live_price": round(live,  6),
                    "stop":       stop,
                    "tp1":        tp1,
                    "tp2":        tp2,
                    "qty":        abs(size),
                    "unrealised": round(upnl, 4),
                    "pnl_pct":    round(pnl_pct, 2),
                    "progress":   round(progress, 1),
                    "confidence": conf,
                    "score":      score,
                    "reasons":    t.get("reasons", []),
                    "tier":       t.get("tier", ""),
                    "opened_at":  t.get("opened_at", ""),
                    "exchange":   "deribit_testnet",
                })
            return jsonify(result)
        except Exception as e:
            log.error(f"Deribit positions: {e}")

    # File fallback — use trades.json with Binance prices
    trades = ai_data
    result = []
    for symbol, t in trades.items():
        if t.get("closed"):
            continue
        entry = float(t.get("entry", 0) or 0)
        live  = live_prices.get(symbol, 0.0)
        qty   = float(t.get("qty", 0) or 0)
        sig   = t.get("signal", "BUY")
        upnl  = ((live - entry) * qty if sig == "BUY" else (entry - live) * qty) if live and entry else 0
        pct   = ((live - entry) / entry * 100 if sig == "BUY" else (entry - live) / entry * 100) if entry else 0
        result.append({**t, "symbol": symbol, "live_price": live,
                       "unrealised": round(upnl, 4), "pnl_pct": round(pct, 2), "progress": 0})
    return jsonify(result)

# ── /api/trades/history ─────────────────────────────────────────────────
@app.route("/api/trades/history")
def api_trade_history():
    bust("trade_history.json")
    h    = get("trade_history.json", [])
    real = [x for x in h if x.get("signal") != "RECOVERED"]
    return jsonify(list(reversed(real[-100:])))

# ── /api/signals ────────────────────────────────────────────────────────
@app.route("/api/signals")
def api_signals():
    bust("signals.json")
    sigs     = get("signals.json", [])
    symbol   = request.args.get("symbol")
    sig_type = request.args.get("type")
    limit    = int(request.args.get("limit", 100))
    if symbol:   sigs = [s for s in sigs if s.get("symbol") == symbol]
    if sig_type: sigs = [s for s in sigs if s.get("signal") == sig_type.upper()]
    return jsonify(list(reversed(sigs[-limit:])))

# ── /api/log ─────────────────────────────────────────────────────────────
@app.route("/api/log")
def api_log():
    bust("bot.log")
    content = get("bot.log", "")
    if not content:
        return jsonify({"log": "No log entries. Bot runs via GitHub Actions.", "lines": 0})
    lines = content.splitlines(keepends=True)[-200:]
    return jsonify({"log": "".join(lines), "lines": len(lines)})

# ── /api/market ──────────────────────────────────────────────────────────
@app.route("/api/market")
def api_market():
    symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT",
               "LINKUSDT","NEARUSDT","DOTUSDT","ADAUSDT","INJUSDT","ARBUSDT",
               "OPUSDT","UNIUSDT","AAVEUSDT","FETUSDT","RENDERUSDT","SEIUSDT","SUIUSDT","APTUSDT"]
    prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=10)
        if r.ok:
            for item in r.json():
                if item["symbol"] in symbols:
                    prices[item["symbol"]] = {
                        "price":      float(item.get("lastPrice", 0)),
                        "change_24h": float(item.get("priceChangePercent", 0)),
                        "volume_24h": float(item.get("quoteVolume", 0)),
                        "high_24h":   float(item.get("highPrice", 0)),
                        "low_24h":    float(item.get("lowPrice", 0)),
                    }
    except Exception as e:
        log.warning(f"Market: {e}")
    return jsonify(prices)

# ── /api/scan ─────────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST"])
def api_scan():
    if not GH_TOKEN or not GH_REPO:
        return jsonify({"error": "GH_PAT_TOKEN not configured"}), 400
    headers = {"Authorization": f"token {GH_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json"}
    for wf in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf}/dispatches",
                headers=headers,
                json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}},
                timeout=15
            )
            if r.status_code in (200, 204):
                for f in ["trades.json","balance.json","signals.json","bot.log"]:
                    bust(f)
                return jsonify({"status": "triggered",
                                "message": "Scan started — results appear in ~60s"})
        except Exception as e:
            log.warning(f"wf {wf}: {e}")
    return jsonify({"error": "Could not trigger scan — check GH_PAT_TOKEN"}), 500

# ── /api/performance ──────────────────────────────────────────────────────
@app.route("/api/performance")
def api_performance():
    bust("trade_history.json")
    h    = get("trade_history.json", [])
    real = [x for x in h if x.get("signal") != "RECOVERED"]
    wins = [x for x in real if (x.get("pnl") or 0) > 0]
    loss = [x for x in real if (x.get("pnl") or 0) <= 0]
    tpnl = sum(x.get("pnl", 0) for x in real)
    by_symbol, daily = {}, {}
    for x in real:
        sym = x.get("symbol", "?")
        if sym not in by_symbol: by_symbol[sym] = {"trades":0,"wins":0,"pnl":0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"]    += x.get("pnl", 0)
        if (x.get("pnl") or 0) > 0: by_symbol[sym]["wins"] += 1
        day = (x.get("closed_at") or x.get("opened_at",""))[:10]
        if day: daily[day] = round(daily.get(day, 0) + x.get("pnl", 0), 4)
    lt = sum(x["pnl"] for x in loss)
    return jsonify({
        "total_trades": len(real), "wins": len(wins), "losses": len(loss),
        "win_rate":     round(len(wins)/len(real)*100, 1) if real else 0,
        "total_pnl":    round(tpnl, 4),
        "avg_win":      round(sum(x["pnl"] for x in wins)/len(wins), 4) if wins else 0,
        "avg_loss":     round(sum(x["pnl"] for x in loss)/len(loss), 4) if loss else 0,
        "profit_factor":round(abs(sum(x["pnl"] for x in wins)/lt), 2) if lt else 0,
        "by_symbol": by_symbol, "daily_pnl": daily,
    })

# ── /api/config ───────────────────────────────────────────────────────────
@app.route("/api/config")
def api_config():
    return jsonify({
        "max_open_trades": 3, "risk_per_trade_pct": 2.0,
        "atr_stop_mult": 1.5, "atr_tp1_mult": 2.0, "atr_tp2_mult": 3.0,
        "min_confidence_active": 50, "min_confidence_quiet": 55,
        "exchange": "Deribit Testnet (USDC Linear Perpetuals)",
        "model_accuracy": 73.1,
    })

# ── /api/close_trade ──────────────────────────────────────────────────────
@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    symbol = (request.get_json() or {}).get("symbol")
    if not symbol: return jsonify({"error": "symbol required"}), 400
    bust("trades.json")
    trades = get("trades.json", {})
    if symbol not in trades: return jsonify({"error": f"{symbol} not found"}), 404
    trade = trades.pop(symbol)
    for p in [Path("trades.json"), Path("data/trades.json")]:
        try: p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(trades, indent=2))
        except Exception: pass
    bust("trades.json")
    return jsonify({"status": "removed", "symbol": symbol,
                    "warning": "Also close on Deribit UI!"})

# ── /api/sync ─────────────────────────────────────────────────────────────
@app.route("/api/sync")
def api_sync():
    for f in ["trades.json","trade_history.json","signals.json","balance.json",
              "scan_mode.json","bot.log"]:
        bust(f)
    return jsonify({"status": "synced"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard starting — port {port} | repo {GH_REPO}")
    app.run(host="0.0.0.0", port=port, debug=False)
