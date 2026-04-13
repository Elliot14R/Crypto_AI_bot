# deribit_client.py — Master Production Version
import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.001, "tick_size": 0.1},
    "ETHUSDT":    {"instrument": "ETH_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.01,  "tick_size": 0.01},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.0001},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.1,   "tick_size": 0.01},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.0001},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.0001},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 0.1,   "tick_size": 0.01},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.0001},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.0001},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.0001},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,     "tick_size": 0.001},
}

TRADEABLE_SYMBOLS: list = []


class DeribitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id          = client_id
        self.client_secret      = client_secret
        self.base               = TESTNET_BASE
        self.session            = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "CryptoBotAI/3.0"})
        self._token_expiry      = 0
        self._instrument_cache  = {}
        self._supported_symbols = set()
        self._authenticate()
        self._verify_instruments()

    # ── Auth ──────────────────────────────────────────────────────────

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
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            return {}
        if "error" in data:
            raise Exception(f"Deribit error GET {path}: {data['error']}")
        r.raise_for_status()
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        """JSON-RPC 2.0 Wrapper with safe error extraction to catch 11050 and 10003 details"""
        self._ensure_auth()
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": path.strip("/"),
            "params": body
        }
        r = self.session.post(f"{self.base}{path}", json=payload, timeout=15)
        
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            return {}
            
        if "error" in data:
            err = data["error"]
            raise Exception(f"{err.get('message', err)} (Code: {err.get('code', 'Unknown')})")
            
        r.raise_for_status()
        return data.get("result", data)

    # ── Instrument verification ───────────────────────────────────────

    def _verify_instruments(self):
        """Check active symbols using bulk requests to prevent rate-limiting"""
        global TRADEABLE_SYMBOLS
        confirmed = []
        active_instruments = {}

        for cur in ["USDC", "BTC", "ETH"]:
            try:
                res = self._get("/public/get_instruments", {"currency": cur, "kind": "future", "expired": "false"})
                if isinstance(res, list):
                    for inst in res:
                        name = inst.get("instrument_name")
                        if name: active_instruments[name] = inst
            except Exception as e:
                log.warning(f"Failed to fetch {cur} master list: {e}")

        for sym, info in SYMBOL_MAP.items():
            target_inst = info["instrument"]
            if target_inst in active_instruments:
                self._instrument_cache[target_inst] = active_instruments[target_inst]
                self._supported_symbols.add(sym)
                confirmed.append(sym)
            else:
                log.warning(f"  ⚠️ {sym} ({target_inst}) is currently offline on Deribit Testnet.")

        TRADEABLE_SYMBOLS = confirmed
        log.info(f"✓ Tradeable: {len(confirmed)} symbols — {confirmed}")

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
            self._instrument_cache[name] = self._get("/public/get_instrument", {"instrument_name": name})
        return self._instrument_cache[name]

    def get_tick_size(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        return float(info.get("tick_size") or SYMBOL_MAP[symbol].get("tick_size", 0.001))

    def get_min_trade_amount(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        api_min = info.get("min_trade_amount")
        fallback = SYMBOL_MAP[symbol].get("min_amount", 1.0)
        return float(api_min) if api_min else float(fallback)

    def round_price(self, symbol: str, price: float) -> float:
        if price <= 0: return 0.0
        tick = self.get_tick_size(symbol)
        rounded = round(round(price / tick) * tick, 10)
        decimals = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return round(rounded, decimals)

    def to_int_amount(self, symbol: str, raw_amount: float):
        """Converts to step size correctly, with ZeroDivisionError protection"""
        if raw_amount <= 0:
            return 0.0
            
        min_amt = self.get_min_trade_amount(symbol)
        if not min_amt or min_amt <= 0:
            min_amt = 1.0

        steps  = math.floor(raw_amount / min_amt)
        result = max(1, steps) * min_amt
        
        decimals = len(str(min_amt).rstrip("0").split(".")[-1]) if "." in str(min_amt) else 0
        return round(result, decimals) if decimals > 0 else int(result)

    # ── Position sizing ───────────────────────────────────────────────

    def calc_contracts(self, symbol: str, balance_usd: float, entry: float, stop: float, risk_mult: float = 1.0):
        try:
            from config import RISK_PER_TRADE as risk_pct
        except ImportError:
            risk_pct = 0.01

        risk_usd  = balance_usd * risk_pct * risk_mult
        stop_dist = abs(entry - stop)
        min_amt   = self.get_min_trade_amount(symbol)

        if stop_dist <= 0:
            return self.to_int_amount(symbol, min_amt)

        # Linear: risk_usd = contracts × stop_dist
        raw = risk_usd / stop_dist
        max_contracts = (balance_usd * 0.20) / entry
        raw = min(raw, max_contracts)

        result = self.to_int_amount(symbol, raw)
        log.info(f"  Contracts: {result} {symbol} (risk=${risk_usd:.2f}, kind=linear, stop_dist={stop_dist:.4f}, raw={raw:.2f})")
        return result

    def split_amount(self, symbol: str, total) -> tuple:
        """ZeroDivisionError protected split"""
        if total <= 0:
            return 0, 0
            
        min_amt = self.get_min_trade_amount(symbol)
        if not min_amt or min_amt <= 0:
            min_amt = 1.0

        half = total / 2.0
        tp1_amt = max(min_amt, math.floor(half / min_amt) * min_amt)
        tp2_amt = total - tp1_amt

        if tp2_amt < min_amt:
            tp1_amt = total
            tp2_amt = 0

        decimals = len(str(min_amt).rstrip("0").split(".")[-1]) if "." in str(min_amt) else 0
        if decimals > 0:
            return round(tp1_amt, decimals), round(tp2_amt, decimals)
        return int(tp1_amt), int(tp2_amt)

    # ── Balance ───────────────────────────────────────────────────────

    def get_account_summary(self, currency: str) -> dict:
        return self._get("/private/get_account_summary", {"currency": currency, "extended": "true"})

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                s  = self.get_account_summary(currency)
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
                r = self._get("/private/get_positions", {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    positions.extend(p for p in r if float(p.get("size", 0) or 0) != 0)
            return positions
        except Exception as e:
            log.warning(f"  get_positions: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, amount) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result = self._post(method, {
            "instrument_name": instrument,
            "amount":          amount,
            "type":            "market",
            "time_in_force":   "immediate_or_cancel", # 🟢 IOC FIX: Prevents wash-trading overlaps on testnet
            "label":           f"bot_entry_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ MARKET {side.upper()} {amount} {instrument} id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result 

    def get_fill_price(self, market_result: dict, fallback: float) -> float:
        trades = market_result.get("trades", [])
        if trades:
            prices = [float(t["price"]) for t in trades if t.get("price")]
            if prices: return round(sum(prices) / len(prices), 8)
        order = market_result.get("order", market_result)
        avg   = order.get("average_price") or order.get("price")
        if avg and float(avg) > 0: return float(avg)
        return fallback

    def place_limit_order(self, symbol: str, side: str, amount, price: float, stop_price: float = None) -> dict:
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
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} @ {safe_price} id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
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
                r = self._get("/private/get_open_orders_by_instrument", {"instrument_name": self.get_instrument_name(symbol)})
                return r if isinstance(r, list) else []
            orders = []
            for cur in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_open_orders_by_currency", {"currency": cur})
                orders.extend(r if isinstance(r, list) else [])
            return orders
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def get_live_price(self, symbol: str) -> float:
        try:
            t = self._get("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)})
            return float(t.get("mark_price") or t.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  price {symbol}: {e}")
            return 0.0

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit Testnet — ${total:.2f} USD | {len(TRADEABLE_SYMBOLS)} symbols")
            return True
        except Exception as e:
            log.error(f"✗ Deribit FAILED: {e}")
            raise
