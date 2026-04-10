# deribit_client.py — Precision Rounding Fix
import json, time, logging, requests, math
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# ── Complete Deribit instrument map ──────────────────────────────────
SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",      "currency": "BTC",  "kind": "inverse", "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",      "currency": "ETH",  "kind": "inverse", "min_amount": 1},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
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

TRADEABLE_SYMBOLS = list(SYMBOL_MAP.keys())

class DeribitClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.base          = TESTNET_BASE
        self.session       = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._access_token = None
        self._token_expiry = 0
        self._instrument_cache = {}
        self._authenticate()

    def _authenticate(self):
        r = self.session.get(
            f"{self.base}/public/auth",
            params={"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        res  = data.get("result", {})
        if not res or "access_token" not in res:
            raise Exception(f"Auth failed: {data}")
        self._access_token = res["access_token"]
        self._token_expiry = time.time() + res.get("expires_in", 900) - 60
        self.session.headers["Authorization"] = f"Bearer {self._access_token}"
        log.info("✓ Deribit testnet authenticated")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry: self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data: raise Exception(f"API error {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data: raise Exception(f"API error {path}: {data['error']}")
        return data.get("result", data)

    def is_supported(self, symbol: str) -> bool:
        if symbol not in SYMBOL_MAP: return False
        instrument = SYMBOL_MAP[symbol]["instrument"]
        try:
            if instrument not in self._instrument_cache:
                info = self._get("/public/get_instrument", {"instrument_name": instrument})
                self._instrument_cache[instrument] = info
            return True
        except Exception: return False

    def get_instrument_name(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP: raise ValueError(f"{symbol} not supported.")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_instrument_info(self, symbol: str) -> dict:
        name = self.get_instrument_name(symbol)
        if name not in self._instrument_cache:
            info = self._get("/public/get_instrument", {"instrument_name": name})
            self._instrument_cache[name] = info
        return self._instrument_cache.get(name, {})

    def get_min_trade_amount(self, symbol: str) -> float:
        info = self.get_instrument_info(symbol)
        return float(info.get("min_trade_amount", SYMBOL_MAP.get(symbol, {}).get("min_amount", 1)))

    def get_live_price(self, symbol: str) -> float:
        try:
            instrument = self.get_instrument_name(symbol)
            ticker     = self._get("/public/ticker", {"instrument_name": instrument})
            return float(ticker.get("mark_price") or ticker.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  Deribit price {symbol}: {e}")
            return 0.0

    def round_amount(self, symbol: str, amount: float) -> float:
        """Strictly rounds an amount to the exact exchange step size to prevent 400 Errors."""
        if amount <= 0: return 0.0
        min_amt = self.get_min_trade_amount(symbol)
        
        # Calculate how many "steps" fit into the requested amount
        steps = math.floor(amount / min_amt)
        rounded = steps * min_amt
        
        # Ensure it meets the absolute minimum trade size
        final_amount = max(min_amt, rounded)
        
        # Format cleanly for the API payload (remove .0 if it's an integer step)
        if float(min_amt).is_integer():
            return int(final_amount)
        else:
            dec = len(str(float(min_amt)).split('.')[-1])
            return round(final_amount, dec)

    def calc_contracts(self, symbol: str, balance_usd: float, entry: float, stop: float, risk_mult: float = 1.0) -> float:
        risk_usd  = balance_usd * 0.02 * risk_mult   # 2% risk
        stop_dist = abs(entry - stop)
        min_amt   = self.get_min_trade_amount(symbol)

        if stop_dist <= 0: return self.round_amount(symbol, min_amt)

        info = SYMBOL_MAP.get(symbol, {})
        kind = info.get("kind", "linear")

        if kind == "inverse":
            pnl_per_usd_move = 10.0 / entry
            amount = risk_usd / (stop_dist * pnl_per_usd_move)
        else:
            amount     = risk_usd / stop_dist
            max_amount = balance_usd * 0.20 / entry
            amount     = min(amount, max_amount)

        final_amount = self.round_amount(symbol, amount)
        log.info(f"  Contracts: {final_amount} {symbol} (risk=${risk_usd:.2f}, kind={kind}, step={min_amt})")
        return final_amount

    def get_account_summary(self, currency: str) -> dict:
        try: return self._get("/private/get_account_summary", {"currency": currency, "extended": "true"})
        except Exception: return {}

    def get_all_balances(self) -> dict:
        balances = {}
        for currency in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                summary = self.get_account_summary(currency)
                eq_usd  = float(summary.get("equity_usd", 0) or summary.get("equity", 0) or 0)
                avail   = float(summary.get("available_funds", 0) or 0)
                if eq_usd > 0:
                    balances[currency] = {"equity_usd": round(eq_usd, 2), "available":  round(avail, 6)}
            except Exception as e: log.debug(f"  Balance {currency}: {e}")
        return balances

    def get_total_equity_usd(self) -> float:
        balances = self.get_all_balances()
        return round(sum(v.get("equity_usd", 0) for v in balances.values()), 2)

    def get_positions(self) -> list:
        try:
            positions = []
            for currency in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_positions", {"currency": currency, "kind": "future"})
                if isinstance(r, list): positions.extend([p for p in r if float(p.get("size",0) or 0) != 0])
            return positions
        except Exception as e:
            log.warning(f"  Positions: {e}"); return []

    def place_market_order(self, symbol: str, side: str, amount: float) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result     = self._post(method, {
            "instrument_name": instrument, "amount": amount, "type": "market",
            "label": f"bot_{symbol}_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  Market {side.upper()} {amount} {instrument} → id={order.get('order_id','?')}")
        return order

    def place_limit_order(self, symbol: str, side: str, amount: float, price: float, stop_price: float = None) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        body = {
            "instrument_name": instrument, "amount": amount, "price": round(price, 4),
            "label": f"bot_{symbol}_{int(time.time())}",
        }
        if stop_price is not None:
            body["type"] = "stop_limit"
            body["stop_price"] = round(stop_price, 4)
            body["trigger"] = "last_price"
        else: body["type"] = "limit"

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  {kind} {side.upper()} {amount} {instrument} @ {price} → id={order.get('order_id','?')}")
        return order

    def get_order(self, order_id: str) -> dict:
        try: return self._get("/private/get_order_state", {"order_id": order_id})
        except Exception as e: log.warning(f"  get_order {order_id}: {e}"); return {}

    def cancel_order(self, order_id: str) -> dict:
        try: return self._post("/private/cancel", {"order_id": order_id})
        except Exception as e: log.warning(f"  cancel {order_id}: {e}"); return {}

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit Testnet — portfolio ${total:.2f} USD")
            return True
        except Exception as e:
            log.error(f"✗ Deribit connection failed: {e}")
            raise
