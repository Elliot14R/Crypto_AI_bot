# deribit_client.py — FULL INTEGRATED VERSION (Math + Connection Restored)
import time, logging, requests, math
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",      "currency": "BTC",  "kind": "inverse", "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",      "currency": "ETH",  "kind": "inverse", "min_amount": 1},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
}
TRADEABLE = list(SYMBOL_MAP.keys())
TRADEABLE_SYMBOLS = TRADEABLE  # Alias to prevent import errors

class DeribitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base = TESTNET_BASE
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._access_token = None
        self._token_expiry = 0
        self._instrument_cache = {}
        self._authenticate()

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={
            "grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret
        }, timeout=15)
        res = r.json().get("result")
        if not res: raise Exception(f"Auth failed: {r.text}")
        self._access_token = res["access_token"]
        self._token_expiry = time.time() + res.get("expires_in", 900) - 60
        self.session.headers["Authorization"] = f"Bearer {self._access_token}"
        log.info("✓ Deribit testnet authenticated")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry: self._authenticate()

    def _get(self, path, params=None):
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        try: data = r.json()
        except: raise Exception(f"HTTP {r.status_code}: {r.text}")
        if "error" in data:
            err = data["error"].get("message", data["error"])
            raise Exception(f"API Error ({path}): {err}")
        return data.get("result", data)

    def _post(self, path, body):
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        try: data = r.json()
        except: raise Exception(f"HTTP {r.status_code}: {r.text}")
        if "error" in data:
            err = data["error"].get("message", data["error"])
            raise Exception(f"API Error ({path}): {err}")
        return data.get("result", data)

    def get_instrument_info(self, symbol: str) -> dict:
        inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
        if inst not in self._instrument_cache:
            info = self._get("/public/get_instrument", {"instrument_name": inst})
            self._instrument_cache[inst] = info
        return self._instrument_cache.get(inst, {})

    # ── PRECISION & MATH RESTORED ─────────────────────────────────────
    
    def get_min_trade_amount(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        return float(info.get("min_trade_amount", SYMBOL_MAP.get(symbol, {}).get("min_amount", 1)))

    def round_amount(self, symbol: str, amount: float) -> float:
        if amount <= 0: return 0.0
        step = self.get_min_trade_amount(symbol)
        rounded = max(step, math.floor(amount / step) * step)
        return int(rounded) if float(step).is_integer() else round(rounded, 6)

    def round_price(self, symbol: str, price: float) -> float:
        info = self.get_instrument_info(symbol)
        tick = float(info.get("tick_size", 0.01))
        return round(round(price / tick) * tick, 8)

    def calc_contracts(self, symbol: str, balance_usd: float, entry: float, stop: float, risk_mult: float = 1.0) -> float:
        try:
            from config import RISK_PER_TRADE as rpt
        except ImportError:
            rpt = 0.01
        risk_usd = balance_usd * rpt * risk_mult
        stop_dist = abs(entry - stop)
        min_amt = self.get_min_trade_amount(symbol)

        if stop_dist <= 0: return self.round_amount(symbol, min_amt)

        kind = SYMBOL_MAP.get(symbol, {}).get("kind", "linear")
        if kind == "inverse":
            amount = risk_usd / (stop_dist / entry)
        else:
            amount = min(risk_usd / stop_dist, balance_usd * 0.20 / entry)

        final_amount = self.round_amount(symbol, amount)
        log.info(f"  Contracts: {final_amount} {symbol} (risk=${risk_usd:.2f}, {kind})")
        return final_amount

    # ──────────────────────────────────────────────────────────────────

    def get_live_price(self, symbol: str) -> float:
        try:
            inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
            ticker = self._get("/public/ticker", {"instrument_name": inst})
            return float(ticker.get("mark_price") or ticker.get("last_price") or 0)
        except Exception: return 0.0

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH", "USDC"]:
            try:
                res = self._get("/private/get_account_summary", {"currency": currency, "extended": "true"})
                eq_usd = float(res.get("equity_usd", 0) or res.get("equity", 0) or 0)
                avail = float(res.get("available_funds", 0) or 0)
                if eq_usd > 0 or avail > 0:
                    balances[currency] = {"equity_usd": round(eq_usd, 2), "available": round(avail, 6)}
            except: continue
        return balances

    def get_usdt_equivalent(self) -> float:
        balances = self.get_all_balances()
        return round(sum(v.get("equity_usd", 0) for v in balances.values()), 2)

    def get_total_equity_usd(self) -> float:
        return self.get_usdt_equivalent()

    def get_positions(self) -> list:
        positions = []
        for currency in ["BTC", "ETH", "USDC"]:
            try:
                r = self._get("/private/get_positions", {"currency": currency, "kind": "future"})
                if isinstance(r, list): positions.extend([p for p in r if float(p.get("size",0)) != 0])
            except: continue
        return positions

    def place_market_order(self, symbol, side, amount):
        inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
        method = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        return self._post(method, {"instrument_name": inst, "amount": amount, "type": "market"})

    def place_limit_order(self, symbol, side, amount, price, stop_price=None):
        inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
        method = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        safe_price = self.round_price(symbol, price)
        body = {"instrument_name": inst, "amount": amount, "price": safe_price, "reduce_only": True}
        if stop_price:
            body.update({"type": "stop_limit", "stop_price": self.round_price(symbol, stop_price), "trigger": "last_price"})
        else:
            body["type"] = "limit"
        return self._post(method, body)

    def get_order(self, order_id):
        return self._get("/private/get_order_state", {"order_id": str(order_id)})

    def is_order_filled(self, order: dict) -> bool:
        return order.get("order_state") == "filled"

    def get_fill_price(self, order_result: dict, fallback: float) -> float:
        trades = order_result.get("trades", [])
        if trades:
            prices = [float(t.get("price", 0)) for t in trades if t.get("price")]
            if prices: return round(sum(prices) / len(prices), 2)
        order = order_result.get("order", order_result)
        avg = order.get("average_price") or order.get("price")
        return float(avg) if avg and float(avg) > 0 else fallback

    def is_supported(self, symbol):
        return symbol in SYMBOL_MAP

    def cancel_order(self, order_id: str) -> dict:
        return self._post("/private/cancel", {"order_id": order_id})

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit Testnet — ${total:.2f} USD | {len(TRADEABLE)} symbols")
            return True
        except Exception as e:
            log.error(f"✗ Deribit FAILED: {e}")
            raise
