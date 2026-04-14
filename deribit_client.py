import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

class DeribitClient:
    def __init__(self, client_id, client_secret):
        self.client_id, self.client_secret = client_id, client_secret
        self.base, self.session = TESTNET_BASE, requests.Session()
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
            # 🟢 FIX: Print the exact payload so we NEVER have to guess what caused the error again
            raise Exception(f"Deribit: {data['error'].get('message')} | Payload: {body}")
        return data.get("result", data)

    def _get(self, path, params=None):
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        return r.json().get("result", {})

    def _verify_instruments(self):
        res = self._get("/public/get_instruments", {"currency": "USDC", "kind": "future"})
        active = {i["instrument_name"]: i for i in res}
        from config import SYMBOLS
        for sym in SYMBOLS:
            inst = f"{sym.replace('USDT', '')}_USDC-PERPETUAL"
            if inst in active:
                self._instrument_cache[inst] = active[inst]
                self._supported_symbols.add(sym)
        log.info(f"✓ Verified {len(self._supported_symbols)} USDC symbols")

    def test_connection(self):
        total = self.get_total_equity_usd()
        log.info(f"✅ Deribit Live: ${total:.2f}")
        return True

    def get_total_equity_usd(self):
        return float(self._get("/private/get_account_summary", {"currency": "USDC"}).get("equity", 0))

    def get_tradeable(self): return list(self._supported_symbols)
    def is_supported(self, symbol): return symbol in self._supported_symbols
    def get_instrument_name(self, symbol): return f"{symbol.replace('USDT', '')}_USDC-PERPETUAL"

    def round_amount(self, symbol, raw):
        inst = self.get_instrument_name(symbol)
        step = float(self._instrument_cache[inst].get("min_trade_amount", 1.0))
        precision = len(str(step).split('.')[1].rstrip('0')) if '.' in str(step) else 0
        res = max(step, math.floor(raw / step) * step)
        clean_val = "{: .{}f}".format(res, precision).strip()
        return float(clean_val) if precision > 0 else int(float(clean_val))

    def round_price(self, symbol, price):
        inst = self.get_instrument_name(symbol)
        tick = float(self._instrument_cache[inst].get("tick_size", 0.01))
        precision = len(str(tick).split('.')[1].rstrip('0')) if '.' in str(tick) else 0
        res = round(round(price / tick) * tick, precision)
        clean_val = "{: .{}f}".format(res, precision).strip()
        return float(clean_val) if precision > 0 else int(float(clean_val))

    def calc_contracts(self, symbol, balance_usd, entry, stop, risk_mult=1.0):
        raw = (balance_usd * 0.01 * risk_mult) / abs(entry - stop)
        return self.round_amount(symbol, min(raw, (balance_usd * 0.2) / entry))

    def split_amount(self, symbol, total):
        q1 = self.round_amount(symbol, total / 2)
        return q1, self.round_amount(symbol, total - q1)

    def get_positions(self): 
        return [p for p in self._get("/private/get_positions", {"currency": "USDC"}) if float(p.get("size", 0)) != 0]

    def place_market_order(self, symbol, side, amount):
        inst = self.get_instrument_name(symbol)
        # 🟢 FIX: Removed 'time_in_force'. Deribit violently rejects this for Market orders.
        payload = {
            "instrument_name": inst, 
            "amount": self.round_amount(symbol, amount), 
            "type": "market"
        }
        return self._post(f"/private/{side.lower()}", payload)

    def place_limit_order(self, symbol, side, amount, price, stop_price=None):
        inst = self.get_instrument_name(symbol)
        payload = {
            "instrument_name": inst, 
            "amount": self.round_amount(symbol, amount), 
            "price": self.round_price(symbol, price), 
            "reduce_only": True
        }
        if stop_price: 
            payload.update({"type": "stop_limit", "trigger_price": self.round_price(symbol, stop_price), "trigger": "last_price"})
        else: 
            payload["type"] = "limit"
        return self._post(f"/private/{side.lower()}", payload)

    def get_order(self, oid): return self._post("/private/get_order_state", {"order_id": str(oid)})
    def is_order_filled(self, o): return o.get("order_state") == "filled"
