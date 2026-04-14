# deribit_client.py — FINAL FIX
#
# ROOT CAUSE OF -32602 INVALID PARAMS:
#   Previous _post() wrapped requests in JSON-RPC 2.0 format:
#   {"jsonrpc":"2.0","method":"private/buy","params":{...}}
#   This format is ONLY for Deribit WebSocket API.
#   REST API endpoint /api/v2/private/buy wants the body DIRECTLY.
#   Removing the wrapper fixes all -32602 errors instantly.

import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# All USDC linear perpetuals — confirmed on Deribit testnet
SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.001, "tick_size": 0.1},
    "ETHUSDT":    {"instrument": "ETH_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.01,  "tick_size": 0.01},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.01},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "min_amount": 0.1,   "tick_size": 0.01},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "min_amount": 1,     "tick_size": 0.0001},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL", "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",  "currency": "USDC", "min_amount": 1,     "tick_size": 0.001},
}

# Populated at runtime after verifying with Deribit API
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

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={
            "grant_type":    "client_credentials",
            "client_id":     self.client_id,
            "client_secret": self.client_secret,
        }, timeout=15)
        r.raise_for_status()
        res = r.json().get("result", {})
        if not res or "access_token" not in res:
            raise Exception(f"Deribit auth failed: {r.text[:200]}")
        self.session.headers["Authorization"] = f"Bearer {res['access_token']}"
        self._token_expiry = time.time() + int(res.get("expires_in", 900)) - 60
        log.info("✓ Deribit testnet authenticated")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        """HTTP GET for public endpoints and read-only private endpoints."""
        self._ensure_auth()
        r    = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        data = r.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code", "") if isinstance(err, dict) else ""
            raise Exception(f"{msg} (Code: {code})" if code else msg)
        r.raise_for_status()
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        """
        HTTP POST for trading/private write endpoints.
        
        CRITICAL: Deribit REST API takes the body DIRECTLY as JSON.
        Do NOT wrap in {"jsonrpc":"2.0","method":...,"params":...} —
        that envelope format is ONLY for the WebSocket API.
        Wrapping it causes -32602 Invalid params on every order.
        """
        self._ensure_auth()
        r    = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        data = r.json()
        if "error" in data:
            err  = data["error"]
            msg  = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            code = err.get("code", "") if isinstance(err, dict) else ""
            raise Exception(f"{msg} (Code: {code})" if code else msg)
        r.raise_for_status()
        return data.get("result", data)

    # ── Instrument verification ───────────────────────────────────────

    def _verify_instruments(self):
        """Bulk-load all live instruments to verify which are available on testnet."""
        global TRADEABLE_SYMBOLS
        active = {}
        for currency in ["USDC", "BTC", "ETH"]:
            try:
                res = self._get("/public/get_instruments",
                                {"currency": currency, "kind": "future", "expired": "false"})
                if isinstance(res, list):
                    for inst in res:
                        name = inst.get("instrument_name")
                        if name:
                            active[name] = inst
            except Exception as e:
                log.warning(f"  Instrument list {currency}: {e}")

        confirmed = []
        for sym, info in SYMBOL_MAP.items():
            target = info["instrument"]
            if target in active:
                self._instrument_cache[target] = active[target]
                self._supported_symbols.add(sym)
                confirmed.append(sym)
            else:
                log.warning(f"  ⚠️ {sym} ({target}) offline on testnet — skip")

        TRADEABLE_SYMBOLS = confirmed
        log.info(f"✓ Tradeable: {len(confirmed)} — {confirmed}")

    # ── Symbol helpers ────────────────────────────────────────────────

    def is_supported(self, symbol: str) -> bool:
        return symbol in self._supported_symbols

    def get_instrument_name(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP:
            raise ValueError(f"{symbol} not in SYMBOL_MAP")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_instrument_info(self, symbol: str) -> dict:
        name = self.get_instrument_name(symbol)
        if name not in self._instrument_cache:
            self._instrument_cache[name] = self._get(
                "/public/get_instrument", {"instrument_name": name}
            )
        return self._instrument_cache[name]

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
        """Round to valid step size. Returns float or int depending on step."""
        if raw <= 0: return 0.0
        step     = self.get_min_trade_amount(symbol)
        steps    = math.floor(raw / step)
        result   = max(step, steps * step)
        decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        return round(result, decimals) if decimals else int(result)

    def split_amount(self, symbol: str, total) -> tuple:
        """Split total into two valid halves (tp1, tp2)."""
        if total <= 0: return 0, 0
        step     = self.get_min_trade_amount(symbol)
        half     = total / 2.0
        tp1      = max(step, math.floor(half / step) * step)
        tp2      = total - tp1
        if tp2 < step:
            tp1 = total
            tp2 = 0
        decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
        if decimals:
            return round(tp1, decimals), round(tp2, decimals)
        return int(tp1), int(tp2)

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}")
            return 0.0

    # ── Position sizing ───────────────────────────────────────────────

    def calc_contracts(self, symbol: str, balance_usd: float,
                       entry: float, stop: float, risk_mult: float = 1.0):
        """
        2% risk per trade, capped at 20% of balance.
        Linear perp: amount = base currency units (e.g. 5.0 SOL, 2.5 DOT)
        PnL per unit per $1 price move = 1 USDC
        """
        try:
            from config import RISK_PER_TRADE as risk_pct
        except ImportError:
            risk_pct = 0.02

        risk_usd  = balance_usd * risk_pct * risk_mult
        stop_dist = abs(entry - stop)
        min_amt   = self.get_min_trade_amount(symbol)

        if stop_dist <= 0 or entry <= 0:
            return self.round_amount(symbol, min_amt)

        # Units risked = risk_usd / stop_dist_in_usd
        raw = risk_usd / stop_dist
        # Cap: position value ≤ 20% of balance
        max_pos = (balance_usd * 0.20) / entry
        raw     = min(raw, max_pos)

        result = self.round_amount(symbol, raw)
        log.info(f"  Contracts: {result} {symbol} "
                 f"(risk=${risk_usd:.2f} stop_dist={stop_dist:.4f} raw={raw:.2f})")
        return result

    # ── Balance ───────────────────────────────────────────────────────

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                s  = self._get("/private/get_account_summary",
                               {"currency": currency, "extended": "true"})
                eq = float(s.get("equity_usd") or s.get("equity") or 0)
                av = float(s.get("available_funds", 0) or 0)
                if eq > 0:
                    balances[currency] = {"equity_usd": round(eq, 2), "available": round(av, 6)}
            except Exception as e:
                log.debug(f"  Balance {currency}: {e}")
        return balances

    def get_total_equity_usd(self) -> float:
        return round(sum(v["equity_usd"] for v in self.get_all_balances().values()), 2)

    def get_positions(self) -> list:
        try:
            positions = []
            for currency in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_positions",
                              {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    positions.extend(
                        p for p in r if float(p.get("size", 0) or 0) != 0
                    )
            return positions
        except Exception as e:
            log.warning(f"  get_positions: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, amount) -> dict:
        """Market order with IOC — returns result dict containing 'order' and 'trades'."""
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result     = self._post(method, {
            "instrument_name": instrument,
            "amount":          amount,
            "type":            "market",
            "time_in_force":   "immediate_or_cancel",
            "label":           f"bot_entry_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ MARKET {side.upper()} {amount} {instrument} "
                 f"id={order.get('order_id','')} state={order.get('order_state','')}")
        return result

    def get_fill_price(self, market_result: dict, fallback: float) -> float:
        """Extract actual fill price from market order result."""
        trades = market_result.get("trades", [])
        if trades:
            prices = [float(t["price"]) for t in trades if t.get("price")]
            if prices:
                return round(sum(prices) / len(prices), 8)
        order = market_result.get("order", market_result)
        avg   = order.get("average_price") or order.get("price")
        if avg and float(avg) > 0:
            return float(avg)
        return fallback

    def place_limit_order(self, symbol: str, side: str, amount,
                          price: float, stop_price: float = None) -> dict:
        """
        Limit or stop-limit order.
        Uses 'trigger_price' field (not 'stop_price') for stop-limit orders.
        All prices rounded to valid tick size.
        reduce_only=True ensures these only close existing positions.
        """
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)

        if stop_price is not None:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "stop_limit",
                "price":           safe_price,
                "trigger_price":   self.round_price(symbol, stop_price),
                "trigger":         "last_price",
                "reduce_only":     True,
                "label":           f"bot_sl_{int(time.time())}",
            }
        else:
            body = {
                "instrument_name": instrument,
                "amount":          amount,
                "type":            "limit",
                "price":           safe_price,
                "reduce_only":     True,
                "label":           f"bot_tp_{int(time.time())}",
            }

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} "
                 f"@ {safe_price} id={order.get('order_id','')} state={order.get('order_state','')}")
        return result

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            if "order_not_found" in str(e).lower():
                return {"order_state": "not_found"}
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def is_order_filled(self, order: dict) -> bool:
        return order.get("order_state", "").lower() == "filled"

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self._post("/private/cancel", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  cancel {order_id}: {e}")
            return {}

    def get_open_orders(self, symbol: str = None) -> list:
        try:
            if symbol and symbol in SYMBOL_MAP:
                r = self._get("/private/get_open_orders_by_instrument",
                              {"instrument_name": self.get_instrument_name(symbol)})
                return r if isinstance(r, list) else []
            orders = []
            for cur in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_open_orders_by_currency", {"currency": cur})
                orders.extend(r if isinstance(r, list) else [])
            return orders
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit Testnet OK — ${total:.2f} USD | "
                     f"{len(TRADEABLE_SYMBOLS)} tradeable symbols")
            return True
        except Exception as e:
            log.error(f"✗ Deribit connection failed: {e}")
            raise
