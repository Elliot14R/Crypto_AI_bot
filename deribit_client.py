# deribit_client.py — Ghost trade fix: broader order fill detection
import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
        # Big 3
        "BTCUSDT":    {"instrument": "BTC_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.001, "tick_size": 0.1},
        "ETHUSDT":    {"instrument": "ETH_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.01,  "tick_size": 0.01},
        "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.001},
        
        # High-Liquidity L1s
        "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
        "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
        "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.001},
        "MATICUSDT":  {"instrument": "MATIC_USDC-PERPETUAL","currency": "USDC", "min_amount": 10,    "tick_size": 0.0001},
        "ATOMUSDT":   {"instrument": "ATOM_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        
        # Institutional Alts
        "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.01},
        "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
        "LTCUSDT":    {"instrument": "LTC_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.01},
        "BCHUSDT":    {"instrument": "BCH_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.01,  "tick_size": 0.05},
        
        # AI & Momentum
        "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
        "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL", "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.001},
        "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
        "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
        "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
        "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    }
TRADEABLE_SYMBOLS: list = []


class DeribitClient:

    def __init__(self, client_id: str, client_secret: str):
        self.client_id          = client_id
        self.client_secret      = client_secret
        self.base               = TESTNET_BASE
        self.session            = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._token_expiry      = 0
        self._instrument_cache  = {}
        self._supported_symbols = set()
        self._authenticate()
        self._verify_instruments()

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={
            "grant_type": "client_credentials",
            "client_id": self.client_id, "client_secret": self.client_secret,
        }, timeout=15)
        r.raise_for_status()
        res = r.json().get("result", {})
        if not res or "access_token" not in res:
            raise Exception(f"Auth failed: {r.text[:200]}")
        self.session.headers["Authorization"] = f"Bearer {res['access_token']}"
        self._token_expiry = time.time() + int(res.get("expires_in", 900)) - 60
        log.info("✓ Deribit testnet authenticated")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r    = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        data = r.json()
        if "error" in data:
            err  = data["error"]
            msg  = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code", "")          if isinstance(err, dict) else ""
            raise Exception(f"{msg} (Code:{code})" if code else msg)
        r.raise_for_status()
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        """Plain HTTP POST — NO JSON-RPC wrapper (that's WebSocket only)."""
        self._ensure_auth()
        r    = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        data = r.json()
        if "error" in data:
            err  = data["error"]
            msg  = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code", "")          if isinstance(err, dict) else ""
            raise Exception(f"{msg} (Code:{code})" if code else msg)
        r.raise_for_status()
        return data.get("result", data)

    def _verify_instruments(self):
        global TRADEABLE_SYMBOLS
        active = {}
        try:
            res = self._get("/public/get_instruments",
                            {"currency": "USDC", "kind": "future", "expired": "false"})
            if isinstance(res, list):
                active = {i["instrument_name"]: i for i in res}
        except Exception as e:
            log.warning(f"  Instrument list: {e}")
        confirmed = []
        for sym, info in SYMBOL_MAP.items():
            target = info["instrument"]
            if target in active:
                self._instrument_cache[target] = active[target]
                self._supported_symbols.add(sym)
                confirmed.append(sym)
        TRADEABLE_SYMBOLS = confirmed
        log.info(f"✓ Tradeable: {len(confirmed)} — {confirmed}")

    def is_supported(self, symbol: str) -> bool:
        return symbol in self._supported_symbols

    def get_tradeable(self) -> list:
        return list(self._supported_symbols)

    def get_instrument_name(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP:
            raise ValueError(f"{symbol} not in SYMBOL_MAP")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_instrument_info(self, symbol: str) -> dict:
        name = self.get_instrument_name(symbol)
        if name not in self._instrument_cache:
            self._instrument_cache[name] = self._get(
                "/public/get_instrument", {"instrument_name": name})
        return self._instrument_cache.get(name, {})

    def get_tick_size(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        return float(info.get("tick_size") or SYMBOL_MAP[symbol].get("tick_size", 0.001))

    def get_min_trade_amount(self, symbol: str) -> float:
        info    = self.get_instrument_info(symbol)
        api_min = info.get("min_trade_amount")
        return float(api_min) if api_min else float(SYMBOL_MAP[symbol].get("min_amount", 1.0))

    def round_price(self, symbol: str, price: float) -> float:
        if price <= 0: return 0.0
        tick     = self.get_tick_size(symbol)
        rounded  = round(round(price / tick) * tick, 10)
        decimals = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return round(rounded, decimals)

    def round_amount(self, symbol: str, raw: float) -> float:
        if raw <= 0: return 0.0
        step     = self.get_min_trade_amount(symbol)
        steps    = math.floor(raw / step)
        result   = max(step, steps * step)
        decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        return round(result, decimals) if decimals else int(result)

    def split_amount(self, symbol: str, total) -> tuple:
        if total <= 0: return 0, 0
        step  = self.get_min_trade_amount(symbol)
        tp1   = max(step, math.floor(total / 2.0 / step) * step)
        tp2   = total - tp1
        if tp2 < step: tp1 = total; tp2 = 0
        decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        return (round(tp1, decimals), round(tp2, decimals)) if decimals else (int(tp1), int(tp2))

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}"); return 0.0

    def calc_contracts(self, symbol: str, balance_usd: float,
                       entry: float, stop: float, risk_mult: float = 1.0):
        try:
            from config import RISK_PER_TRADE as rpt
        except ImportError:
            rpt = 0.01
        risk_usd  = balance_usd * rpt * risk_mult
        stop_dist = abs(entry - stop)
        min_amt   = self.get_min_trade_amount(symbol)
        if stop_dist <= 0 or entry <= 0:
            return self.round_amount(symbol, min_amt)
        raw    = risk_usd / stop_dist
        max_p  = (balance_usd * 0.20) / entry
        raw    = min(raw, max_p)
        result = self.round_amount(symbol, raw)
        log.info(f"  Contracts: {result} {symbol} (risk=${risk_usd:.2f})")
        return result

    def get_all_balances(self) -> dict:
        balances = {}
        for cur in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                s  = self._get("/private/get_account_summary",
                               {"currency": cur, "extended": "true"})
                eq = float(s.get("equity_usd") or s.get("equity") or 0)
                av = float(s.get("available_funds", 0) or 0)
                if eq > 0:
                    balances[cur] = {"equity_usd": round(eq, 2), "available": round(av, 6)}
            except Exception as e:
                log.debug(f"  Balance {cur}: {e}")
        return balances

    def get_total_equity_usd(self) -> float:
        return round(sum(v["equity_usd"] for v in self.get_all_balances().values()), 2)

    def get_positions(self) -> list:
        try:
            positions = []
            for cur in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_positions",
                              {"currency": cur, "kind": "future"})
                if isinstance(r, list):
                    positions.extend(p for p in r if float(p.get("size", 0) or 0) != 0)
            return positions
        except Exception as e:
            log.warning(f"  get_positions: {e}"); return []

    def place_market_order(self, symbol: str, side: str, amount) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result     = self._post(method, {
            "instrument_name": instrument, "amount": amount,
            "type": "market", "time_in_force": "immediate_or_cancel",
            "label": f"bot_entry_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ MARKET {side.upper()} {amount} {instrument} "
                 f"id={order.get('order_id','')} state={order.get('order_state','')}")
        return result

    def get_fill_price(self, market_result: dict, fallback: float) -> float:
        trades = market_result.get("trades", [])
        if trades:
            prices = [float(t["price"]) for t in trades if t.get("price")]
            if prices: return round(sum(prices) / len(prices), 8)
        order = market_result.get("order", market_result)
        avg   = order.get("average_price") or order.get("price")
        return float(avg) if avg and float(avg) > 0 else fallback

    def place_limit_order(self, symbol: str, side: str, amount,
                          price: float, stop_price: float = None) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)
        if stop_price is not None:
            body = {
                "instrument_name": instrument, "amount": amount,
                "type": "stop_limit", "price": safe_price,
                "trigger_price": self.round_price(symbol, stop_price),
                "trigger": "last_price", "reduce_only": True,
                "label": f"bot_sl_{int(time.time())}",
            }
        else:
            body = {
                "instrument_name": instrument, "amount": amount,
                "type": "limit", "price": safe_price,
                "reduce_only": True, "label": f"bot_tp_{int(time.time())}",
            }
        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "SL" if stop_price else "TP"
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} "
                 f"@ {safe_price} id={order.get('order_id','')} state={order.get('order_state','')}")
        return result

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            if "not_found" in str(e).lower():
                return {"order_state": "not_found"}
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def is_order_filled(self, order: dict) -> bool:
        """
        ROOT CAUSE OF GHOST TRADES: Deribit stop-limit orders that trigger
        can show state='cancelled' with filled_amount > 0.
        We must check ALL these states, not just 'filled'.
        """
        state       = order.get("order_state", "").lower()
        filled_amt  = float(order.get("filled_amount", 0) or 0)
        avg_price   = float(order.get("average_price", 0) or 0)

        # Definitively filled states
        if state == "filled":
            return True

        # Stop-limit orders that triggered: show as cancelled but have fill data
        if state in ("cancelled", "closed") and (filled_amt > 0 or avg_price > 0):
            log.info(f"  Triggered order detected: state={state} "
                     f"filled={filled_amt} avg={avg_price}")
            return True

        return False

    def get_order_fill_price(self, order: dict, fallback: float) -> float:
        """Extract fill price — handles both normal fills and triggered stop fills."""
        avg = order.get("average_price")
        if avg and float(avg) > 0:
            return float(avg)
        # Also check last_price field on triggered stops
        lp = order.get("last_price") or order.get("price")
        if lp and float(lp) > 0:
            return float(lp)
        return fallback

    def get_trade_history_for_instrument(self, symbol: str, count: int = 10) -> list:
        """
        Fetch actual trade fills from Deribit for an instrument.
        Used to recover real PnL when ghost cleaner detects a closed position.
        """
        try:
            instrument = self.get_instrument_name(symbol)
            result     = self._get("/private/get_user_trades_by_instrument", {
                "instrument_name": instrument,
                "count":           count,
                "sorting":         "desc",
            })
            return result if isinstance(result, list) else result.get("trades", [])
        except Exception as e:
            log.warning(f"  Trade history {symbol}: {e}")
            return []

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  cancel {order_id}: {e}"); return {}

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit OK — ${total:.2f} | {len(TRADEABLE_SYMBOLS)} symbols")
            return True
        except Exception as e:
            log.error(f"✗ Deribit: {e}"); raise
