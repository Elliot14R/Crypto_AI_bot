# dashboard.py — Live Deribit Fetching + GitHub Sync
import os, json, subprocess, requests, base64, logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app   = Flask(__name__, static_folder="dashboard_static")
CORS(app)
log   = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

DATA_FILES = ["trades.json","trade_history.json","signals.json",
              "balance.json","scan_mode.json","bot.log"]

GH_TOKEN = os.getenv("GH_PAT_TOKEN","")
GH_REPO  = os.getenv("GITHUB_REPO", "Elliot14R/Crypto_AI_bot")
GH_BRANCH= os.getenv("GITHUB_BRANCH","main")


# ── File helpers ──────────────────────────────────────────────────────

def load_json(filename, default):
    for p in [Path(filename), Path("data") / filename]:
        try:
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        except Exception:
            pass
    return default

def load_log(lines=200):
    for p in [Path("bot.log"), Path("data/bot.log")]:
        try:
            if p.exists():
                with open(p) as f:
                    return f.readlines()[-lines:]
        except Exception:
            pass
    return []

def fetch_live_github_data(filename):
    """Bypasses Render's stale disk and safely decodes the base64 file from GitHub"""
    repo = os.getenv("GITHUB_REPO")
    token = os.getenv("GH_PAT_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not repo or not token: 
        return {}
    
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    import base64
    
    for path in [f"data/{filename}", filename]:
        try:
            r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}", headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                # 🟢 THE FIX: Decode the base64 gibberish into readable JSON!
                content = base64.b64decode(data["content"]).decode('utf-8')
                return json.loads(content)
        except Exception as e:
            pass
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


# ── GitHub sync (restore state files from repo) ───────────────────────

def sync_from_github():
    if not GH_TOKEN or not GH_REPO:
        return
    headers = {"Authorization": f"token {GH_TOKEN}",
                "Accept": "application/vnd.github.v3+json"}
    for fname in ["trades.json","trade_history.json","signals.json",
                  "balance.json","scan_mode.json"]:
        for repo_path in [f"data/{fname}", fname]:
            try:
                r = requests.get(
                    f"https://api.github.com/repos/{GH_REPO}/contents/{repo_path}",
                    headers=headers, timeout=10
                )
                if r.status_code == 200:
                    content = base64.b64decode(r.json()["content"]).decode()
                    # Save to local paths
                    Path(fname).write_text(content)
                    Path("data").mkdir(exist_ok=True)
                    (Path("data") / fname).write_text(content)
                    break
            except Exception:
                pass

# Sync on startup
try:
    sync_from_github()
    log.info("Synced state from GitHub")
except Exception as e:
    log.warning(f"GitHub sync failed: {e}")


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard_static", "index.html")

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
    try:
        return send_from_directory("dashboard_static", path)
    except Exception:
        return send_from_directory("dashboard_static", "index.html")


@app.route("/api/status")
def api_status():
    history    = load_json("trade_history.json", [])
    trades     = load_json("trades.json", {})
    signals    = load_json("signals.json", [])
    scan_mode  = load_json("scan_mode.json", {})
    balance    = load_json("balance.json", {})

    real_hist  = [h for h in history if h.get("signal") != "RECOVERED"]
    wins       = [h for h in real_hist if (h.get("pnl") or 0) > 0]
    total_pnl  = sum(h.get("pnl", 0) for h in real_hist)
    win_rate   = round(len(wins)/len(real_hist)*100, 1) if real_hist else 0

    # Today's signals
    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_sigs = [s for s in signals
                  if s.get("generated_at","").startswith(today)]
    buys       = sum(1 for s in today_sigs if s.get("signal")=="BUY" and not s.get("rejected"))
    sells      = sum(1 for s in today_sigs if s.get("signal")=="SELL" and not s.get("rejected"))

    return jsonify({
        "win_rate":     win_rate,
        "total_pnl":    round(total_pnl, 4),
        "wins":         len(wins),
        "losses":       len(real_hist) - len(wins),
        "total_trades": len(real_hist),
        "open_trades":  len(trades),
        "max_trades":   3,
        "scan_mode":    scan_mode.get("mode", "active"),
        "mode_label":   scan_mode.get("mode","active").upper(),
        "today_signals":len(today_sigs),
        "today_buys":   buys,
        "today_sells":  sells,
        "model_accuracy": 73.1,
        "balance":      balance.get("usdt", 0),
        "exchange":     balance.get("exchange", "Deribit Testnet"),
        "last_updated": balance.get("updated_at",""),
    })


@app.route("/api/balance")
def api_balance():
    """Fetches LIVE balance directly from Deribit Testnet!"""
    try:
        client_id = os.getenv("DERIBIT_CLIENT_ID", "")
        client_secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
        if client_id and client_secret:
            from deribit_client import DeribitClient
            client = DeribitClient(client_id, client_secret)
            all_bals = client.get_all_balances()
            
            # Grab the USDC balance for the headline number
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
                "exchange":    "Deribit(by Coinbase) Testnet"
            })
    except Exception as e:
        log.warning(f"Live balance failed: {e}")

    # Fallback to local file if API fails
    bal = load_json("balance.json", {})
    return jsonify({"ok": True, "usdt": bal.get("usdt", 0), "assets": bal.get("assets", [])})


@app.route("/api/trades/open")
def api_open_trades():
    """Fetches LIVE positions from Deribit and stitches AI targets (SL/TP/Conf) from GitHub API"""
    try:
        # Fetch the real AI data (Confidence, Score, SL, TP) from GitHub to fix $0.00 issue
        local_trades = fetch_live_github_data("trades.json")
        
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

                # Correct percentage math (reverses for SELL positions)
                if entry > 0:
                    pct = (live - entry) / entry * 100 if signal == "BUY" else (entry - live) / entry * 100
                else:
                    pct = 0.0

                # Correct USD PNL math (Handles missing floating_profit_loss_usd on Testnet)
                upnl = float(p.get("floating_profit_loss_usd") or 0)
                if upnl == 0:
                    base_pnl = float(p.get("floating_profit_loss") or 0)
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


@app.route("/api/trades/history")
def api_trade_history():
    history = load_json("trade_history.json", [])
    real    = [h for h in history if h.get("signal") != "RECOVERED"]
    return jsonify(list(reversed(real[-100:])))


@app.route("/api/signals")
def api_signals():
    signals  = load_json("signals.json", [])
    symbol   = request.args.get("symbol")
    sig_type = request.args.get("type")
    limit    = int(request.args.get("limit", 50))

    filtered = signals
    if symbol:
        filtered = [s for s in filtered if s.get("symbol") == symbol]
    if sig_type:
        filtered = [s for s in filtered if s.get("signal") == sig_type.upper()]

    return jsonify(list(reversed(filtered[-limit:])))


@app.route("/api/log")
def api_log():
    lines = load_log(200)
    return jsonify({"log": "".join(lines), "lines": len(lines)})


@app.route("/api/market")
def api_market():
    """Live prices for all 20 monitored coins."""
    symbols = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT",
        "XRPUSDT","LINKUSDT","NEARUSDT","DOTUSDT","ADAUSDT",
        "INJUSDT","ARBUSDT","OPUSDT","UNIUSDT","AAVEUSDT",
        "FETUSDT","RENDERUSDT","SEIUSDT","SUIUSDT","APTUSDT"
    ]
    prices = {}
    try:
        r = requests.get(
            "https://data-api.binance.vision/api/v3/ticker/24hr",
            timeout=10
        )
        if r.ok:
            for item in r.json():
                if item["symbol"] in symbols:
                    prices[item["symbol"]] = {
                        "price":        float(item.get("lastPrice", 0)),
                        "change_24h":   float(item.get("priceChangePercent", 0)),
                        "volume_24h":   float(item.get("quoteVolume", 0)),
                        "high_24h":     float(item.get("highPrice", 0)),
                        "low_24h":      float(item.get("lowPrice", 0)),
                    }
    except Exception as e:
        log.warning(f"Market data: {e}")
    return jsonify(prices)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    Trigger GitHub Actions scan.
    Fixed: uses correct workflow_dispatch API format.
    """
    if not GH_TOKEN or not GH_REPO:
        return jsonify({"error": "GH_PAT_TOKEN or GITHUB_REPO not configured"}), 400

    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
    }

    # Find the workflow file
    for workflow in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{workflow}/dispatches",
                headers=headers,
                json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}},
                timeout=15
            )
            if r.status_code in (204, 200):
                log.info(f"✅ GitHub Actions triggered via {workflow}")
                return jsonify({
                    "status":  "triggered",
                    "workflow": workflow,
                    "message": "Scan started on GitHub Actions (takes ~30s to appear)"
                })
        except Exception as e:
            log.warning(f"Workflow {workflow}: {e}")

    return jsonify({"error": "Could not trigger GitHub Actions — check GH_PAT_TOKEN"}), 500


@app.route("/api/performance")
def api_performance():
    history = load_json("trade_history.json", [])
    real    = [h for h in history if h.get("signal") != "RECOVERED"]

    wins     = [h for h in real if (h.get("pnl") or 0) > 0]
    losses   = [h for h in real if (h.get("pnl") or 0) <= 0]
    total_pnl = sum(h.get("pnl", 0) for h in real)
    avg_win   = sum(h["pnl"] for h in wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(h["pnl"] for h in losses) / len(losses) if losses else 0

    # PnL by symbol
    by_symbol = {}
    for h in real:
        sym = h.get("symbol","?")
        if sym not in by_symbol:
            by_symbol[sym] = {"trades":0,"wins":0,"pnl":0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"]    += h.get("pnl",0)
        if (h.get("pnl") or 0) > 0:
            by_symbol[sym]["wins"] += 1

    # Daily PnL
    daily = {}
    for h in real:
        day = (h.get("closed_at") or h.get("opened_at",""))[:10]
        if day:
            daily[day] = daily.get(day, 0) + h.get("pnl", 0)

    return jsonify({
        "total_trades":  len(real),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins)/len(real)*100, 1) if real else 0,
        "total_pnl":     round(total_pnl, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "profit_factor": round(abs(sum(h["pnl"] for h in wins) /
                           sum(h["pnl"] for h in losses)), 2)
                          if losses and sum(h["pnl"] for h in losses) != 0 else 0,
        "by_symbol":     by_symbol,
        "daily_pnl":     daily,
    })


@app.route("/api/config")
def api_config():
    return jsonify({
        "max_open_trades":   3,
        "risk_per_trade_pct": 2.0,
        "atr_stop_mult":     1.5,
        "atr_tp1_mult":      2.0,
        "atr_tp2_mult":      3.0,
        "min_confidence_active":  50,
        "min_confidence_quiet":   55,
        "min_score_active":       1,
        "min_score_quiet":        2,
        "min_adx":               15,
        "scan_interval_min":     15,
        "symbols":               20,
        "exchange":              "Deribit Testnet (USDC Perpetuals)",
        "model_accuracy":        73.1,
    })


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    """
    Manual close — removes trade from trades.json.
    Actual position on Deribit must be closed manually on the exchange UI.
    """
    data   = request.get_json() or {}
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    trades = load_json("trades.json", {})
    if symbol not in trades:
        return jsonify({"error": f"{symbol} not in open trades"}), 404

    trade = trades.pop(symbol)

    # Record as manually closed
    history = load_json("trade_history.json", [])
    history.append({
        **trade,
        "close_price":  trade.get("entry", 0),
        "pnl":          0.0,
        "closed_at":    datetime.now(timezone.utc).isoformat(),
        "close_reason": "Manual close via dashboard",
    })

    # Save both
    for p in [Path("trades.json"), Path("data/trades.json")]:
        try:
            Path(p).parent.mkdir(exist_ok=True)
            with open(p,"w") as f: json.dump(trades, f, indent=2)
        except Exception: pass
    for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
        try:
            Path(p).parent.mkdir(exist_ok=True)
            with open(p,"w") as f: json.dump(history, f, indent=2, default=str)
        except Exception: pass

    return jsonify({
        "status":  "removed",
        "symbol":  symbol,
        "warning": "Close the actual position on Deribit testnet UI too!"
    })


@app.route("/api/sync")
def api_sync():
    """Force re-sync state files from GitHub."""
    try:
        sync_from_github()
        return jsonify({"status": "synced", "message": "State files refreshed from GitHub"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"CryptoBot Dashboard starting on port {port}")
    log.info(f"GitHub repo: {GH_REPO} | Branch: {GH_BRANCH}")
    app.run(host="0.0.0.0", port=port, debug=False)
