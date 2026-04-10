# deribit_client.py — Deribit Testnet Client
# ══════════════════════════════════════════════════════════════════════
# WHY DERIBIT:
#   ✅ IP whitelisting is OPTIONAL — works from any IP (GitHub Actions OK)
#   ✅ No geo-block (unlike Binance/Bybit for India)
#   ✅ Free testnet at test.deribit.com — no KYC needed
#   ✅ BTC/ETH perpetuals with USDC margin
#   ✅ REST API with simple client_id + client_secret auth
#
# SETUP (one time):
#   1. Register at https://test.deribit.com (free, no KYC)
#   2. Click "Deposit" → get 10 BTC or 10 ETH of test funds
#   3. Go to Settings → API → Create API key
#      Scopes: trade:read_write  account:read  wallet:read
#      Leave IP Whitelist EMPTY (works from any IP)
#   4. Add DERIBIT_CLIENT_ID and DERIBIT_CLIENT_SECRET to GitHub Secrets
#
# INSTRUMENTS: BTC-PERPETUAL, ETH-PERPETUAL (USD-settled)
# ACCOUNT TYPE: standard BTC/ETH margin account
# ══════════════════════════════════════════════════════════════════════

import json
import time
import logging
import requests

log = logging.getLogger(__name__)

TESTNET_BASE = "https://test.deribit.com/api/v2"

# Map our USDT symbols → Deribit instruments + currency
SYMBOL_MAP = {
    "BTCUSDT":  {"instrument": "BTC-PERPETUAL",  "currency": "BTC"},
    "ETHUSDT":  {"instrument": "ETH-PERPETUAL",  "currency": "ETH"},
    # Add more as Deribit adds linear USDC perpetuals
}

# Symbols we can actually trade on Deribit testnet
TRADEABLE = list(SYMBOL_MAP.keys())


class DeribitClient:
    """
    Minimal Deribit testnet client.
    Uses client_credentials grant (no IP whitelist needed).
    Trades BTC-PERPETUAL and ETH-PERPETUAL.
    """

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.base          = TESTNET_BASE
        self.session       = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._access_token = None
        self._token_expiry = 0
        self._authenticate()

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self):
        """Get OAuth2 access token using client_credentials."""
        try:
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
            if data.get("result"):
                self._access_token = data["result"]["access_token"]
                self._token_expiry = time.time() + data["result"].get("expires_in", 900) - 60
                self.session.headers["Authorization"] = f"Bearer {self._access_token}"
                log.info("✓ Deribit testnet authenticated")
            else:
                raise Exception(f"Auth failed: {data}")
        except Exception as e:
            raise Exception(f"Deribit auth failed: {e}")

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _get(self, path: str, params: dict = None) -> dict:
        self._ensure_auth()
        r = self.session.get(f"{self.base}{path}", params=params or {}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"GET {path}: {data['error']}")
        return data.get("result", data)

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_auth()
        r = self.session.post(f"{self.base}{path}", json=body, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise Exception(f"POST {path}: {data['error']}")
        return data.get("result", data)

    # ── Symbol helpers ────────────────────────────────────────────────

    def get_instrument(self, symbol: str) -> str:
        """Get Deribit instrument name for a symbol."""
        if symbol not in SYMBOL_MAP:
            raise ValueError(f"{symbol} not supported on Deribit. Supported: {TRADEABLE}")
        return SYMBOL_MAP[symbol]["instrument"]

    def get_currency(self, symbol: str) -> str:
        return SYMBOL_MAP[symbol]["currency"]

    def is_supported(self, symbol: str) -> bool:
        return symbol in SYMBOL_MAP

    # ── Balance ───────────────────────────────────────────────────────

    def get_balance(self, currency: str = "BTC") -> dict:
        """Get account summary for a currency (BTC or ETH)."""
        return self._get("/private/get_account_summary", {"currency": currency, "extended": true})

    def get_all_balances(self) -> dict:
        """Returns equity in USD for BTC and ETH accounts combined."""
        balances = {}
        for currency in ["BTC", "ETH"]:
            try:
                summary = self.get_balance(currency)
                equity_usd = float(summary.get("equity_usd", 0) or
                                   summary.get("equity", 0) or 0)
                balances[currency] = {
                    "equity_usd": round(equity_usd, 2),
                    "currency":   currency,
                    "available":  float(summary.get("available_funds", 0) or 0),
                }
            except Exception as e:
                log.warning(f"Balance {currency}: {e}")
        return balances

    def get_usdt_equivalent(self) -> float:
        """Return total portfolio value in USD across BTC + ETH accounts."""
        balances = self.get_all_balances()
        total    = sum(v.get("equity_usd", 0) for v in balances.values())
        return round(total, 2)

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, amount_usd: float) -> dict:
        """
        Place a market order.
        amount_usd: notional USD value of position (e.g. 100 = $100 of BTC)
        Deribit BTC-PERPETUAL: amount is in USD contracts ($10 per contract)
        """
        instrument = self.get_instrument(symbol)
        # BTC-PERPETUAL min = $10. Round to nearest 10.
        contracts  = max(10, round(amount_usd / 10) * 10)

        method = "private/buy" if side.upper() == "BUY" else "private/sell"
        result = self._post(f"/{method}", {
            "instrument_name": instrument,
            "amount":          contracts,
            "type":            "market",
            "label":           f"bot_{symbol}_{int(time.time())}",
        })
        order = result.get("order", result)
        log.info(f"  ✅ DERIBIT MARKET {side.upper()} ${contracts} {instrument} "
                 f"— id:{order.get('order_id','')}")
        return order

    def place_limit_order(self, symbol: str, side: str, amount_usd: float,
                          price: float, stop_price: float = None) -> dict:
        """Place limit or stop-limit order."""
        instrument = self.get_instrument(symbol)
        contracts  = max(10, round(amount_usd / 10) * 10)

        if stop_price:
            # Stop-limit order
            method = "private/buy" if side.upper() == "BUY" else "private/sell"
            result = self._post(f"/{method}", {
                "instrument_name": instrument,
                "amount":          contracts,
                "type":            "stop_limit",
                "price":           round(price, 2),
                "stop_price":      round(stop_price, 2),
                "trigger":         "last_price",
                "label":           f"sl_{symbol}_{int(time.time())}",
            })
        else:
            method = "private/buy" if side.upper() == "BUY" else "private/sell"
            result = self._post(f"/{method}", {
                "instrument_name": instrument,
                "amount":          contracts,
                "type":            "limit",
                "price":           round(price, 2),
                "label":           f"tp_{symbol}_{int(time.time())}",
            })

        order = result.get("order", result)
        kind  = "STOP_LIMIT" if stop_price else "LIMIT"
        log.info(f"  ✅ DERIBIT {kind} {side.upper()} ${contracts} {instrument} @ {price} "
                 f"— id:{order.get('order_id','')}")
        return order

    def get_order(self, order_id: str) -> dict:
        """Get order status."""
        try:
            return self._get("/private/get_order_state", {"order_id": order_id})
        except Exception as e:
            log.warning(f"  get_order {order_id}: {e}")
            return {}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order."""
        try:
            return self._post("/private/cancel", {"order_id": order_id})
        except Exception as e:
            log.warning(f"  cancel_order {order_id}: {e}")
            return {}

    def get_open_orders(self, symbol: str = None) -> list:
        """Get open orders."""
        try:
            if symbol and symbol in SYMBOL_MAP:
                instrument = self.get_instrument(symbol)
                result     = self._get("/private/get_open_orders_by_instrument",
                                       {"instrument_name": instrument})
            else:
                result = []
                for currency in ["BTC", "ETH"]:
                    r = self._get("/private/get_open_orders_by_currency",
                                  {"currency": currency})
                    result.extend(r if isinstance(r, list) else [])
            return result if isinstance(result, list) else []
        except Exception as e:
            log.warning(f"  get_open_orders: {e}")
            return []

    def get_position(self, symbol: str) -> dict:
        """Get current position for a symbol."""
        try:
            instrument = self.get_instrument(symbol)
            return self._get("/private/get_position", {"instrument_name": instrument})
        except Exception:
            return {}

    def get_live_price(self, symbol: str) -> float:
        """Get current mark price for a symbol."""
        try:
            instrument = self.get_instrument(symbol)
            ticker     = self._get("/public/ticker", {"instrument_name": instrument})
            return float(ticker.get("mark_price", ticker.get("last_price", 0)) or 0)
        except Exception as e:
            log.warning(f"  Deribit price {symbol}: {e}")
            return 0.0

    def calc_usd_amount(self, balance_usd: float, entry: float, stop: float,
                        risk_mult: float = 1.0) -> float:
        """
        Calculate USD position size based on 1% risk rule.
        Returns the USD notional amount to trade (rounded to $10).
        """
        risk_usd   = balance_usd * 0.01 * risk_mult
        stop_dist  = abs(entry - stop) / entry   # as fraction
        if stop_dist <= 0: return 10.0
        # USD amount = risk / stop_distance_fraction
        amount_usd = risk_usd / stop_dist
        # Cap at 20% of balance
        amount_usd = min(amount_usd, balance_usd * 0.20)
        # Round to $10 (Deribit minimum contract size)
        return max(10.0, round(amount_usd / 10) * 10)

    def test_connection(self) -> bool:
        """Test connection and return True if working."""
        try:
            total = self.get_usdt_equivalent()
            log.info(f"✅ Deribit Testnet connected — Portfolio: ~${total:.2f} USD")
            return True
        except Exception as e:
            log.error(f"✗ Deribit connection FAILED: {e}")
            return False
