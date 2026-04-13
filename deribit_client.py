# deribit_client.py — bad_request fix + all previous fixes
#
# FIX FOR bad_request CODE 11050:
#   Deribit requires INTEGER contract amounts for ALL perpetuals.
#   calc_contracts() was returning floats (0.5149 BTC, 16.691 ETH).
#   Now uses int(math.floor(...)) — guaranteed integers every time.
#
# CONTRACT SIZE REFERENCE (Deribit testnet):
#   BTC-PERPETUAL  : inverse, $10/contract, min=10, step=10 → integers (10, 20, 30...)
#   ETH-PERPETUAL  : inverse, $1/contract,  min=1,  step=1  → integers (1, 2, 3...)
#   *_USDC-PERPETUAL: linear, base currency, min=1,  step=1  → integers
#
# ALL OTHER FIXES RETAINED:
#   _post() sends real HTTP POST (not _get)
#   stop_price field correct (trigger_price per Deribit docs for stop-limit)
#   get_fill_price() reads trades[] array
#   is_order_filled() checks order_state == "filled"
#   reduce_only=True on all SL/TP orders
#   _verify_instruments() on startup

import math, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# Contract specs — verified against Deribit documentation
# inverse: amount = USD notional contracts (integer required)
#   BTC: $10 per contract, min 10 contracts
#   ETH: $1 per contract,  min 1 contract
# linear: amount = base currency units (integer required)
SYMBOL_MAP = {
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",       "currency": "BTC",  "kind": "inverse", "contract_usd": 10, "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",       "currency": "ETH",  "kind": "inverse", "contract_usd": 1,  "min_amount": 1},
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "AAVEUSDT":   {"instrument": "AAVE_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "INJUSDT":    {"instrument": "INJ_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "ARBUSDT":    {"instrument": "ARB_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "OPUSDT":     {"instrument": "OP_USDC-PERPETUAL",   "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "SEIUSDT":    {"instrument": "SEI_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "FETUSDT":    {"instrument": "FET_USDC-PERPETUAL",  "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
    "RENDERUSDT": {"instrument": "RNDR_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "contract_usd": 0,  "min_amount": 1},
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
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error GET {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        # CONFIRMED FIX: real HTTP POST, never _get()
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"Deribit error POST {path}: {data['error']}")
        return data.get("result", data)

    # ── Instrument verification ───────────────────────────────────────

    def _verify_instruments(self):
        """Check each symbol live on startup. Only confirmed symbols are tradeable."""
        global TRADEABLE_SYMBOLS
        confirmed = []
        for sym, info in SYMBOL_MAP.items():
            try:
                result = self._get("/public/get_instrument", {"instrument_name": info["instrument"]})
                self._instrument_cache[info["instrument"]] = result
                self._supported_symbols.add(sym)
                confirmed.append(sym)
            except Exception:
                log.debug(f"  {sym} ({info['instrument']}) not on testnet — skipped")
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
        return float(info.get("tick_size") or SYMBOL_MAP[symbol].get("tick_size", 0.5))

    def get_min_trade_amount(self, symbol: str) -> int:
        """Always returns an integer — Deribit min_trade_amount is always integer."""
        info = self.get_instrument_info(symbol)
        api_min = info.get("min_trade_amount")
        fallback = SYMBOL_MAP[symbol]["min_amount"]
        return int(float(api_min)) if api_min else int(fallback)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to tick_size. Prevents invalid_params on price."""
        if price <= 0: return 0.0
        tick = self.get_tick_size(symbol)
        rounded = round(round(price / tick) * tick, 10)
        decimals = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
        return round(rounded, decimals)

    def to_int_amount(self, symbol: str, raw_amount: float) -> int:
        """
        THE KEY FIX: Convert any float amount to a valid integer contract count.
        Deribit ALWAYS requires integer amounts — floats cause bad_request 11050.
        Uses math.floor (not round) to never exceed risk budget.
        """
        min_amt = self.get_min_trade_amount(symbol)
        # floor to nearest valid step
        steps  = math.floor(raw_amount / min_amt)
        result = max(1, steps) * min_amt
        return int(result)

    # ── Position sizing ───────────────────────────────────────────────

    def calc_contracts(self, symbol: str, balance_usd: float, entry: float,
                       stop: float, risk_mult: float = 1.0) -> int:
        """
        Returns INTEGER contract count. No floats. Ever.
        Inverse (BTC, ETH): amount = USD notional contracts
            BTC: each contract = $10 USD, min 10 contracts
            ETH: each contract = $1  USD, min 1 contract
        Linear (USDC perps): amount = base currency units, min 1
        """
        try:
            from config import RISK_PER_TRADE as risk_pct
        except ImportError:
            risk_pct = 0.01

        risk_usd  = balance_usd * risk_pct * risk_mult
        stop_dist = abs(entry - stop)
        spec      = SYMBOL_MAP[symbol]
        min_amt   = self.get_min_trade_amount(symbol)

        if stop_dist <= 0:
            return min_amt

        if spec["kind"] == "inverse":
            cs = spec["contract_usd"]   # $10 for BTC, $1 for ETH
            # risk_usd = contracts × cs × stop_dist / entry
            # → contracts = risk_usd × entry / (cs × stop_dist)
            raw = risk_usd * entry / (cs * stop_dist)
            # hard cap: 20% portfolio notional
            max_contracts = int((balance_usd * 0.20) / cs)
            raw = min(raw, max_contracts)
        else:
            # linear: risk_usd = contracts × stop_dist
            raw = risk_usd / stop_dist
            # hard cap: 20% portfolio value in base currency
            max_contracts = int((balance_usd * 0.20) / entry)
            raw = min(raw, max_contracts)

        result = self.to_int_amount(symbol, raw)
        log.info(f"  Contracts: {result} {symbol} "
                 f"(risk=${risk_usd:.2f}, kind={spec['kind']}, "
                 f"stop_dist={stop_dist:.4f}, raw={raw:.2f})")
        return result

    def split_amount(self, symbol: str, total: int) -> tuple:
        """
        Split total contracts into (tp1_amount, tp2_amount).
        Both must be valid integers. If total < 2, put all in tp1.
        """
        min_amt = self.get_min_trade_amount(symbol)
        if total < 2 * min_amt:
            return int(total), 0
        half    = total // 2
        tp1_amt = max(min_amt, int(math.floor(half / min_amt) * min_amt))
        tp2_amt = total - tp1_amt
        if tp2_amt < min_amt:
            return int(total), 0
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

    def place_market_order(self, symbol: str, side: str, amount: int) -> dict:
        """
        amount MUST be integer. Returns full result dict {"order": {...}, "trades": [...]}.
        """
        assert isinstance(amount, int) and amount > 0, f"amount must be positive int, got {amount!r}"
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result = self._post(method, {
            "instrument_name": instrument,
            "amount":          amount,
            "type":            "market",
            "label":           f"bot_entry_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ MARKET {side.upper()} {amount} {instrument} "
                 f"id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result  # full result — caller uses get_fill_price()

    def get_fill_price(self, market_result: dict, fallback: float) -> float:
        """Extract actual fill price from market order result trades[] array."""
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

    def place_limit_order(self, symbol: str, side: str, amount: int,
                          price: float, stop_price: float = None) -> dict:
        """
        amount MUST be integer.
        Returns full result dict {"order": {...}} so caller can read order_id.
        reduce_only=True prevents accidental new positions.
        trigger_price field per Deribit docs for stop-limit orders.
        """
        assert isinstance(amount, int) and amount > 0, f"amount must be positive int, got {amount!r}"
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
        log.info(f"  ✅ {kind} {side.upper()} {amount} {instrument} @ {safe_price} "
                 f"id:{order.get('order_id','')} state:{order.get('order_state','')}")
        return result  # full result, caller extracts result["order"]["order_id"]

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": str(order_id)})
        except Exception as e:
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def is_order_filled(self, order: dict) -> bool:
        """Valid Deribit states: open, filled, cancelled, untriggered, rejected."""
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
