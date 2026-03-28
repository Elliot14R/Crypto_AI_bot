# dashboard_api.py - Complete fixed version
# Fixes:
#   1. Cloud Amnesia - pulls state from GitHub on startup
#   2. Close button NoneType crash - uses actual fill price
#   3. Balance shows equity not just free USDT
#   4. All coin categories in every menu

import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TRADES_FILE  = "trades.json"
LOG_FILE     = "bot.log"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"


# ── Persistence layer ─────────────────────────────────────────
def load_json(path, default):
    """Loads from GitHub first, falls back to local, then default."""
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
    """Saves locally AND pushes to GitHub."""
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


# ── Pull state from GitHub on startup ────────────────────────
def restore_on_startup():
    """Called once when Render starts — restores all data from GitHub."""
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
        r    = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            timeout=8
        )
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

    # Read current scan mode
    mode_data = load_json("scan_mode.json", {})

    return jsonify({
        "ok":           True,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "open_trades":  len(trades),
        "max_trades":   3,
        "total_closed": total,
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     wr,
        "total_pnl":    totpnl,
        "model_acc":    model_acc,
        "today_signals": len(today_sigs),
        "today_buy":    len([s for s in today_sigs if s.get("signal") == "BUY"]),
        "today_sell":   len([s for s in today_sigs if s.get("signal") == "SELL"]),
        "scan_mode":    mode_data.get("mode", "active"),
    })


@app.route("/api/balance")
def api_balance():
    """
    Returns balance + equity.
    Equity = free USDT + unrealised PnL of open trades.
    This fixes the 'balance dropped' confusion.
    """
    try:
        from trade_executor import init_exchange
        ex = init_exchange()
        b  = ex.fetch_balance()

        free_usdt  = float(b.get("USDT", {}).get("free",  0) or 0)
        total_usdt = float(b.get("USDT", {}).get("total", 0) or 0)

        # Calculate unrealised PnL from open trades
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
            {
                "asset": a,
                "free":  str(b[a]["free"]),
                "total": str(b[a]["total"]),
            }
            for a in b
            if isinstance(b[a], dict)
            and float(b[a].get("free", 0) or 0) > 0
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
        log.warning(f"Balance fetch failed: {e}")
        return jsonify({
            "ok":         False,
            "usdt":       None,
            "equity":     None,
            "unrealised": 0,
            "error":      str(e),
        })


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
    rejected = request.args.get("rejected")  # ?rejected=true to see filtered signals

    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol]
    if sig_type:
        signals = [s for s in signals if s.get("signal") == sig_type.upper()]
    if rejected == "false":
        signals = [s for s in signals if not s.get("rejected")]
    elif rejected == "true":
        signals = [s for s in signals if s.get("rejected")]

    return jsonify(list(reversed(signals[-200:])))


@app.route("/api/signals/summary")
def api_signals_summary():
    from config import COIN_CATEGORIES, COIN_META
    signals = load_json(SIGNALS_FILE, [])

    coin_stats = {}
    for s in signals:
        sym = s.get("symbol", "")
        if sym not in coin_stats:
            coin_stats[sym] = {
                "total": 0, "buy": 0, "sell": 0,
                "wins": 0, "losses": 0, "avg_conf": [],
            }
        coin_stats[sym]["total"] += 1
        if s.get("signal") == "BUY":
            coin_stats[sym]["buy"] += 1
        elif s.get("signal") == "SELL":
            coin_stats[sym]["sell"] += 1
        if s.get("result") == "WIN":
            coin_stats[sym]["wins"] += 1
        elif s.get("result") == "LOSS":
            coin_stats[sym]["losses"] += 1
        if s.get("confidence"):
            coin_stats[sym]["avg_conf"].append(s["confidence"])

    result = []
    for category, coins in COIN_CATEGORIES.items():
        for sym in coins:
            meta  = COIN_META.get(sym, {})
            stats = coin_stats.get(sym, {
                "total": 0, "buy": 0, "sell": 0,
                "wins": 0, "losses": 0, "avg_conf": [],
            })
            confs   = stats["avg_conf"]
            decided = stats["wins"] + stats["losses"]
            result.append({
                "symbol":   sym,
                "short":    meta.get("short", sym.replace("USDT", "")),
                "color":    meta.get("color", "#888"),
                "category": category,
                "total":    stats["total"],
                "buy":      stats["buy"],
                "sell":     stats["sell"],
                "wins":     stats["wins"],
                "losses":   stats["losses"],
                "win_rate": round(stats["wins"] / decided * 100, 1) if decided else None,
                "avg_conf": round(sum(confs) / len(confs), 1) if confs else None,
            })

    return jsonify(result)


@app.route("/api/coins")
def api_coins():
    from config import COIN_CATEGORIES, COIN_META, SYMBOLS
    prices  = get_live_prices(SYMBOLS)
    signals = load_json(SIGNALS_FILE, [])

    # Latest signal per coin
    sig_map = {}
    for s in signals[-500:]:
        sym = s.get("symbol")
        if sym and not s.get("rejected"):
            sig_map[sym] = s

    result = []
    for category, coins in COIN_CATEGORIES.items():
        for sym in coins:
            meta       = COIN_META.get(sym, {})
            price_data = prices.get(sym) or {}
            last_sig   = sig_map.get(sym, {})
            result.append({
                "symbol":      sym,
                "name":        meta.get("name",  sym.replace("USDT", "")),
                "short":       meta.get("short", sym.replace("USDT", "")),
                "color":       meta.get("color", "#888888"),
                "category":    category,
                "price":       price_data.get("price"),
                "change_pct":  price_data.get("change_pct"),
                "high_24h":    price_data.get("high"),
                "low_24h":     price_data.get("low"),
                "volume_24h":  price_data.get("volume"),
                "last_signal": last_sig.get("signal"),
                "last_conf":   last_sig.get("confidence"),
                "last_sig_time": last_sig.get("generated_at"),
            })
    return jsonify(result)


@app.route("/api/categories")
def api_categories():
    from config import COIN_CATEGORIES, COIN_META, SYMBOLS
    prices  = get_live_prices(SYMBOLS)
    signals = load_json(SIGNALS_FILE, [])

    sig_map = {}
    for s in signals[-200:]:
        sym = s.get("symbol")
        if sym and not s.get("rejected"):
            sig_map[sym] = s

    result = []
    for category, coins in COIN_CATEGORIES.items():
        cat_coins      = []
        avg_change     = []
        bullish_count  = 0

        for sym in coins:
            meta       = COIN_META.get(sym, {})
            price_data = prices.get(sym) or {}
            change     = price_data.get("change_pct", 0) or 0
            avg_change.append(change)
            if change > 0:
                bullish_count += 1
            last_sig = sig_map.get(sym, {})
            cat_coins.append({
                "symbol":      sym,
                "short":       meta.get("short", sym.replace("USDT", "")),
                "color":       meta.get("color", "#888"),
                "price":       price_data.get("price"),
                "change_pct":  change,
                "last_signal": last_sig.get("signal"),
                "last_conf":   last_sig.get("confidence"),
            })

        avg = round(sum(avg_change) / len(avg_change), 2) if avg_change else 0
        result.append({
            "category":   category,
            "coins":      cat_coins,
            "avg_change": avg,
            "bullish":    bullish_count,
            "bearish":    len(coins) - bullish_count,
            "trend":      "BULLISH" if avg > 0.5 else "BEARISH" if avg < -0.5 else "NEUTRAL",
        })

    return jsonify(result)


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
    """
    FIXED: No longer crashes with NoneType.
    Uses actual fill price from exchange instead of live price for PnL.
    """
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

        # Cancel any pending SL/TP orders
        for key, oid in trade.get("order_ids", {}).items():
            if key != "entry":
                try:
                    ex.cancel_order(oid, symbol)
                    log.info(f"  Cancelled {key} order {oid}")
                except Exception:
                    pass

        # Place closing market order
        side = "sell" if trade["signal"] == "BUY" else "buy"
        close_order = ex.create_order(symbol, "market", side, trade["qty"])

        # ── FIX: Use actual fill price, never None ──
        close_price = (
            float(close_order.get("average") or 0) or
            float(close_order.get("price")   or 0) or
            float(trade.get("entry", 0))  # last resort fallback
        )

        entry = float(trade["entry"])
        qty   = float(trade["qty"])
        pnl   = round(
            (close_price - entry) * qty if trade["signal"] == "BUY"
            else (entry - close_price) * qty,
            4
        )

        # Save to history
        history = load_json(HISTORY_FILE, [])
        history.append({
            **trade,
            "close_price":  close_price,
            "pnl":          pnl,
            "closed_at":    datetime.now(timezone.utc).isoformat(),
            "close_reason": "Manual close",
        })
        save_json(HISTORY_FILE, history)

        # Remove from open trades
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


@app.route("/api/persistence/status")
def api_persistence_status():
    """Shows what data is saved in GitHub."""
    try:
        from persistence import get_stats
        return jsonify({"ok": True, "files": get_stats()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/persistence/restore", methods=["POST"])
def api_persistence_restore():
    """Manually trigger GitHub restore — useful after Render restart."""
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


if __name__ == "__main__":
    # Restore data from GitHub on every startup
    restore_on_startup()
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard API → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
