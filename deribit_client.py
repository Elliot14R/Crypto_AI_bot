# deribit_client.py — FINAL INTEGRATED VERSION
import time, logging, requests, math
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",      "currency": "BTC",  "kind": "inverse", "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",      "currency": "ETH",  "kind": "inverse", "min_amount": 1},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    # Add other symbols as needed following the pattern above
}
TRADEABLE = list(SYMBOL_MAP.keys())

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
        return r.json().get("result", r.json())

    def _post(self, path, body):
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        data = r.json()
        if "error" in data: raise Exception(f"Deribit POST Error: {data['error']}")
        return data.get("result", data)

    # FIX for: 'DeribitClient' object has no attribute 'get_live_price'
    def get_live_price(self, symbol: str) -> float:
        try:
            inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
            if not inst: return 0.0
            ticker = self._get("/public/ticker", {"instrument_name": inst})
            return float(ticker.get("mark_price") or ticker.get("last_price") or 0)
        except Exception as e:
            log.warning(f"Price fetch failed for {symbol}: {e}")
            return 0.0

    # FIX for: 'DeribitClient' object has no attribute 'get_total_equity_usd'
    def get_total_equity_usd(self) -> float:
        balances = self.get_all_balances()
        return round(sum(v.get("equity_usd", 0) for v in balances.values()), 2)

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

    def get_positions(self) -> list:
        positions = []
        for currency in ["BTC", "ETH", "USDC"]:
            try:
                r = self._get("/private/get_positions", {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    positions.extend([p for p in r if float(p.get("size", 0)) != 0])
            except: continue
        return positions

    def place_market_order(self, symbol, side, amount):
        inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
        method = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        return self._post(method, {"instrument_name": inst, "amount": amount, "type": "market"})

    def place_limit_order(self, symbol, side, amount, price, stop_price=None):
        inst = SYMBOL_MAP.get(symbol, {}).get("instrument")
        method = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        body = {"instrument_name": inst, "amount": amount, "price": price, "reduce_only": True}
        if stop_price:
            body.update({"type": "stop_limit", "stop_price": stop_price, "trigger": "last_price"})
        else:
            body["type"] = "limit"
        return self._post(method, body)

    def get_order(self, order_id):
        return self._get("/private/get_order_state", {"order_id": str(order_id)})

    def is_order_filled(self, order: dict) -> bool:
        return order.get("order_state") == "filled"

    def is_supported(self, symbol):
        return symbol in SYMBOL_MAP

    def test_connection(self):
        return self.get_total_equity_usd() > 0
