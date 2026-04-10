# deribit_client.py — Fixed Deribit Testnet Client
# 
# ROOT CAUSE OF NO TRADES:
#   Old SYMBOL_MAP only had 2 coins. 18/20 signals silently failed.
#   Fixed: now maps all supported Deribit perpetuals.
#
# Deribit has TWO types of perpetuals:
#   Inverse (BTC/ETH margined): BTC-PERPETUAL, ETH-PERPETUAL
#   Linear (USDC margined):     SOL_USDC-PERPETUAL, XRP_USDC-PERPETUAL etc
# We use LINEAR (USDC) for all — simpler, one balance, same math.

import json, time, logging, requests
log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# ── Complete Deribit instrument map ──────────────────────────────────
# All USDC linear perpetuals (margin in USDC — simple and consistent)
SYMBOL_MAP = {
    # Inverse perpetuals (BTC/ETH margined — use if you hold BTC/ETH)
    "BTCUSDT":    {"instrument": "BTC-PERPETUAL",      "currency": "BTC",  "kind": "inverse", "min_amount": 10},
    "ETHUSDT":    {"instrument": "ETH-PERPETUAL",      "currency": "ETH",  "kind": "inverse", "min_amount": 1},

    # USDC linear perpetuals (easiest — all margin in USDC)
    "SOLUSDT":    {"instrument": "SOL_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "XRPUSDT":    {"instrument": "XRP_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "BNBUSDT":    {"instrument": "BNB_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "AVAXUSDT":   {"instrument": "AVAX_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "LINKUSDT":   {"instrument": "LINK_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "NEARUSDT":   {"instrument": "NEAR_USDC-PERPETUAL","currency": "USDC", "kind": "linear",  "min_amount": 1},
    "DOTUSDT":    {"instrument": "DOT_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "UNIUSDT":    {"instrument": "UNI_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
    "ADAUSDT":    {"instrument": "ADA_USDC-PERPETUAL", "currency": "USDC", "kind": "linear",  "min_amount": 1},
}

# Which symbols we can actually trade
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
        self._instrument_cache = {}   # cache actual instrument details from API
        self._authenticate()

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self):
        r = self.session.get(
            f"{self.base}/public/auth",
            params={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
            },
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
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r    = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"API error {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_auth()
        r    = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"API error {path}: {data['error']}")
        return data.get("result", data)

    # ── Instrument helpers ────────────────────────────────────────────

    def is_supported(self, symbol: str) -> bool:
        """Check if symbol can be traded on Deribit."""
        if symbol not in SYMBOL_MAP:
            return False
        # Verify instrument actually exists on testnet
        instrument = SYMBOL_MAP[symbol]["instrument"]
        try:
            if instrument not in self._instrument_cache:
                info = self._get("/public/get_instrument",
                                 {"instrument_name": instrument})
                self._instrument_cache[instrument] = info
            return True
        except Exception:
            return False

    def get_instrument_name(self, symbol: str) -> str:
        if symbol not in SYMBOL_MAP:
            raise ValueError(
                f"{symbol} not supported on Deribit.\n"
                f"Supported: {', '.join(TRADEABLE_SYMBOLS)}"
            )
        return SYMBOL_MAP[symbol]["instrument"]

    def get_instrument_info(self, symbol: str) -> dict:
        name = self.get_instrument_name(symbol)
        if name not in self._instrument_cache:
            info = self._get("/public/get_instrument", {"instrument_name": name})
            self._instrument_cache[name] = info
        return self._instrument_cache.get(name, {})

    def get_min_trade_amount(self, symbol: str) -> float:
        """Get minimum order size from API (or fallback to SYMBOL_MAP)."""
        info = self.get_instrument_info(symbol)
        return float(info.get("min_trade_amount",
                     SYMBOL_MAP.get(symbol, {}).get("min_amount", 1)))

    def get_contract_size(self, symbol: str) -> float:
        """Get contract size (how much underlying per 1 contract)."""
        info = self.get_instrument_info(symbol)
        return float(info.get("contract_size", 1.0))

    def get_live_price(self, symbol: str) -> float:
        """Get mark price for a symbol."""
        try:
            instrument = self.get_instrument_name(symbol)
            ticker     = self._get("/public/ticker", {"instrument_name": instrument})
            return float(ticker.get("mark_price") or ticker.get("last_price") or 0)
        except Exception as e:
            log.warning(f"  Deribit price {symbol}: {e}")
            return 0.0

    def calc_contracts(self, symbol: str, balance_usd: float,
                       entry: float, stop: float, risk_mult: float = 1.0) -> float:
        """
        Calculate order amount.
        For LINEAR USDC perps: amount = number of coins (e.g. 1.5 SOL)
        For INVERSE BTC perp: amount = USD value (e.g. 100 = $100 worth)
        """
        risk_usd  = balance_usd * 0.02 * risk_mult   # 2% risk
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return self.get_min_trade_amount(symbol)

        info      = SYMBOL_MAP.get(symbol, {})
        kind      = info.get("kind", "linear")
        min_amt   = self.get_min_trade_amount(symbol)

        if kind == "inverse":
            # BTC-PERPETUAL: amount in USD contracts ($10 each)
            # PnL per contract per $1 move = 1/entry
            pnl_per_usd_move = 10.0 / entry
            amount = risk_usd / (stop_dist * pnl_per_usd_move)
            # Round to nearest $10
            amount = max(min_amt, round(amount / 10) * 10)
        else:
            # Linear USDC: amount in base currency units
            # PnL per unit per $1 move = 1 USDC
            amount    = risk_usd / stop_dist
            # Cap at 20% of balance
            max_amount = balance_usd * 0.20 / entry
            amount     = min(amount, max_amount)
            amount     = max(min_amt, round(amount, 1))

        log.info(f"  Contracts: {amount} {symbol} "
                 f"(risk=${risk_usd:.2f}, kind={kind})")
        return amount

    # ── Balance ───────────────────────────────────────────────────────

    def get_account_summary(self, currency: str) -> dict:
        try:
            return self._get("/private/get_account_summary",
                             {"currency": currency, "extended": "true"})
        except Exception:
            return {}

    def get_all_balances(self) -> dict:
        """Returns {currency: {equity_usd, available}} for all currencies."""
        balances = {}
        for currency in ["BTC", "ETH", "USDC", "USDT"]:
            try:
                summary = self.get_account_summary(currency)
                eq_usd  = float(summary.get("equity_usd", 0) or
                                summary.get("equity", 0) or 0)
                avail   = float(summary.get("available_funds", 0) or 0)
                if eq_usd > 0:
                    balances[currency] = {
                        "equity_usd": round(eq_usd, 2),
                        "available":  round(avail, 6),
                    }
            except Exception as e:
                log.debug(f"  Balance {currency}: {e}")
        return balances

    def get_total_equity_usd(self) -> float:
        """Total portfolio value in USD across all currencies."""
        balances = self.get_all_balances()
        total    = sum(v.get("equity_usd", 0) for v in balances.values())
        return round(total, 2)

    def get_usdc_balance(self) -> float:
        """Available USDC (for linear perpetuals)."""
        summary = self.get_account_summary("USDC")
        return float(summary.get("available_funds", 0) or 0)

    def get_positions(self) -> list:
        """Get all open positions."""
        try:
            positions = []
            for currency in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_positions",
                              {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    positions.extend([p for p in r if float(p.get("size",0) or 0) != 0])
            return positions
        except Exception as e:
            log.warning(f"  Positions: {e}")
            return []

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, amount: float) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        result     = self._post(method, {
            "instrument_name": instrument,
            "amount":          amount,
            "type":            "market",
            "label":           f"bot_{symbol}_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  Market {side.upper()} {amount} {instrument} → id={order.get('order_id','?')}")
        return order

    def place_limit_order(self, symbol: str, side: str, amount: float,
                          price: float, stop_price: float = None) -> dict:
        instrument = self.get_instrument_name(symbol)
        method     = "/private/buy" if side.upper() == "BUY" else "/private/sell"
        body = {
            "instrument_name": instrument,
            "amount":          amount,
            "price":           round(price, 4),
            "label":           f"bot_{symbol}_{int(time.time())}",
        }
        if stop_price is not None:
            body["type"]        = "stop_limit"
            body["stop_price"]  = round(stop_price, 4)
            body["trigger"]     = "last_price"
        else:
            body["type"] = "limit"

        result = self._post(method, body)
        order  = result.get("order", result)
        kind   = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  {kind} {side.upper()} {amount} {instrument} @ {price} → id={order.get('order_id','?')}")
        return order

    def get_order(self, order_id: str) -> dict:
        try:
            return self._get("/private/get_order_state", {"order_id": order_id})
        except Exception as e:
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self._post("/private/cancel", {"order_id": order_id})
        except Exception as e:
            log.warning(f"  cancel {order_id}: {e}")
            return {}

    def get_open_orders(self) -> list:
        try:
            orders = []
            for currency in ["BTC", "ETH", "USDC"]:
                r = self._get("/private/get_open_orders_by_currency",
                              {"currency": currency, "kind": "future"})
                if isinstance(r, list):
                    orders.extend(r)
            return orders
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def test_connection(self) -> bool:
        try:
            total = self.get_total_equity_usd()
            log.info(f"✅ Deribit Testnet — portfolio ${total:.2f} USD")
            # Verify which symbols actually work on testnet
            working = []
            for sym in list(SYMBOL_MAP.keys())[:5]:  # check first 5
                try:
                    self.get_instrument_info(sym)
                    working.append(sym)
                except Exception:
                    pass
            log.info(f"  Working symbols: {working}")
            return True
        except Exception as e:
            log.error(f"✗ Deribit connection failed: {e}")
            raise
