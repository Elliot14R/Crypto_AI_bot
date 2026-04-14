# deribit_client.py — Precision-Safe USDC Unified Version
import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.0001, "tick_size": 0.1},
    "ETHUSDT":    {"instrument": "ETH_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.001,  "tick_size": 0.01},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 0.01,   "tick_size": 0.01},
    "SUIUSDT":    {"instrument": "SUI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "APTUSDT":    {"instrument": "APT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 0.1,    "tick_size": 0.001},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "min_amount": 1,      "tick_size": 0.0001},
}

class DeribitClient:
    def __init__(self, client_id, client_secret):
        self.client_id, self.client_secret = client_id, client_secret
        self.base = TESTNET_BASE
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._token_expiry, self._instrument_cache, self._supported_symbols = 0, {}, set()
        self._authenticate()
        self._verify_instruments()

    def _authenticate(self):
        r = self.session.get(f"{self.base}/public/auth", params={"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret}, timeout=15)
        res = r.json().get("result", {})
        self.session.headers["Authorization"] = f"Bearer {res['access_token']}"
        self._token_expiry = time.time() + int(res.get("expires_in", 900)) - 60

    def _ensure_auth(self):
        if time.time() >= self._token_expiry: self._authenticate()

    def _post(self, path, body):
        self._ensure_auth()
        payload = {"jsonrpc": "2.0", "id": int(time.time()*1000), "method": path.strip("/"), "params": body}
        r = self.session.post(f"{self.base}{path}", json=payload, timeout=15)
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit Error: {data['error'].get('message')} (Code: {data['error'].get('code')})")
        return data.get("result", data)

    def _get(self, path, params=None):
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        return r.json().get("result", {})

    def _verify_instruments(self):
        res = self._get("/public/get_instruments", {"currency": "USDC", "kind": "future", "expired": "false"})
        active = {i["instrument_name"]: i for i in res}
        for sym, info in SYMBOL_MAP.items():
            if info["instrument"] in active:
                self._instrument_cache[info["instrument"]] = active[info["instrument"]]
                self._supported_symbols.add(sym)
        log.info(f"✓ Tradeable: {list(self._supported_symbols)}")

    def round_amount(self, symbol, raw):
        """🟢 CLEAN-ROOM PRECISION: Strips float noise using string formatting"""
        step = float(self._instrument_cache[self.get_instrument_name(symbol)].get("min_trade_amount", 1.0))
        precision = len(str(step).split('.')[1].rstrip('0')) if '.' in str(step) else 0
        res = max(step, math.floor(raw / step) * step)
        return float("{: .{}f}".format(res, precision).strip()) if precision > 0 else int(res)

    def round_price(self, symbol, price):
        tick = float(self._instrument_cache[self.get_instrument_name(symbol)].get("tick_size", 0.01))
        precision = len(str(tick).split('.')[1].rstrip('0')) if '.' in str(tick) else 0
        res = round(round(price / tick) * tick, precision)
        return float("{: .{}f}".format(res, precision).strip()) if precision > 0 else int(res)

    def test_connection(self):
        total = self.get_total_equity_usd()
        log.info(f"✅ Connection Verified: ${total:.2f}"); return True

    def get_tradeable(self): return list(self._supported_symbols)
    def is_supported(self, symbol): return symbol in self._supported_symbols
    def get_instrument_name(self, symbol): return SYMBOL_MAP[symbol]["instrument"]
    def calc_contracts(self, symbol, balance_usd, entry, stop, risk_mult=1.0):
        raw = (balance_usd * 0.01 * risk_mult) / abs(entry - stop)
        return self.round_amount(symbol, min(raw, (balance_usd * 0.2) / entry))

    def split_amount(self, symbol, total):
        q1 = self.round_amount(symbol, total / 2); return q1, self.round_amount(symbol, total - q1)

    def get_total_equity_usd(self):
        s = self._get("/private/get_account_summary", {"currency": "USDC", "extended": "true"})
        return float(s.get("equity", 0))

    def get_positions(self): return [p for p in self._get("/private/get_positions", {"currency": "USDC", "kind": "future"}) if float(p.get("size", 0)) != 0]

    def place_market_order(self, symbol, side, amount):
        return self._post(f"/private/{side.lower()}", {"instrument_name": self.get_instrument_name(symbol), "amount": self.round_amount(symbol, amount), "type": "market", "time_in_force": "immediate_or_cancel"})

    def place_limit_order(self, symbol, side, amount, price, stop_price=None):
        params = {"instrument_name": self.get_instrument_name(symbol), "amount": self.round_amount(symbol, amount), "price": self.round_price(symbol, price), "reduce_only": True}
        if stop_price: params.update({"type": "stop_limit", "trigger_price": self.round_price(symbol, stop_price), "trigger": "last_price"})
        else: params["type"] = "limit"
        return self._post(f"/private/{side.lower()}", params)

    def get_order(self, oid):
        try: return self._post("/private/get_order_state", {"order_id": str(oid)})
        except: return {"order_state": "not_found"}
    def is_order_filled(self, o): return o.get("order_state") == "filled"
    def cancel_order(self, oid): return self._post("/private/cancel", {"order_id": str(oid)})
    def get_live_price(self, symbol): return float(self._post("/public/ticker", {"instrument_name": self.get_instrument_name(symbol)}).get("mark_price", 0))
