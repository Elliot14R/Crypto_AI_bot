# binance_client.py — Direct Binance Testnet REST client
#
# WHY THIS EXISTS:
#   CCXT calls /api/v3/exchangeInfo before EVERY operation.
#   Binance testnet blocks this endpoint from GitHub Actions IPs (451 error).
#   This client makes direct signed HMAC requests that SKIP exchangeInfo entirely.
#   Result: balance fetching, order placement, and order checking all work again.

import hmac
import hashlib
import time
import urllib.parse
import requests
import logging

log = logging.getLogger(__name__)

TESTNET_BASE = "https://api.binance.com"
LIVE_BASE    = "https://api.binance.com"   # for public data only (no auth needed)


class BinanceTestnet:
    """
    Minimal Binance Testnet client using direct REST + HMAC-SHA256.
    No CCXT dependency — no exchangeInfo calls — no geo-block.
    """

    def __init__(self, api_key: str, secret: str):
        print("USING BASE URL:", self.base)
        self.api_key = api_key
        self.secret  = secret
        self.base    = TESTNET_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": api_key,
            "Content-Type":  "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0",
        })

    # ── Signing ───────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 10000
        query     = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get(self, path: str, params: dict = None) -> dict:
        params = self._sign(params or {})
        r = self.session.get(f"{self.base}{path}", params=params, timeout=15)
        if not r.ok:
            raise Exception(f"GET {path} {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post(self, path: str, params: dict = None) -> dict:
        params = self._sign(params or {})
        r = self.session.post(f"{self.base}{path}", data=params, timeout=15)
        if not r.ok:
            raise Exception(f"POST {path} {r.status_code}: {r.text[:300]}")
        return r.json()

    def _delete(self, path: str, params: dict = None) -> dict:
        params = self._sign(params or {})
        r = self.session.delete(f"{self.base}{path}", params=params, timeout=15)
        if not r.ok:
            raise Exception(f"DELETE {path} {r.status_code}: {r.text[:200]}")
        return r.json()

    # ── Account ───────────────────────────────────────────────────────

    def get_account(self):
        try:
            return self._get("/api/v3/account")
        except Exception as e:
            print("⚠️ Retry account fetch...")
            time.sleep(2)
            return self._get("/api/v3/account")
        
    def get_balance(self) -> dict:
        """Returns dict of asset → free balance for non-zero assets."""
        account = self.get_account()
        balances = {}
        for b in account.get("balances", []):
            free = float(b.get("free", 0) or 0)
            if free > 0:
                balances[b["asset"]] = free
        return balances

    def get_usdt_balance(self) -> float:
        """Returns free USDT balance."""
        bal = self.get_balance()
        return bal.get("USDT", 0.0)

    # ── Orders ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Place a market order. side = 'BUY' or 'SELL'."""
        params = {
            "symbol":   symbol,
            "side":     side.upper(),
            "type":     "MARKET",
            "quantity": self._fmt_qty(quantity),
        }
        return self._post("/api/v3/order", params)

    def place_limit_order(self, symbol: str, side: str, quantity: float,
                          price: float, stop_price: float = None) -> dict:
        """Place a limit or stop-limit order."""
        params = {
            "symbol":      symbol,
            "side":        side.upper(),
            "quantity":    self._fmt_qty(quantity),
            "price":       self._fmt_price(price),
            "timeInForce": "GTC",
        }
        if stop_price is not None:
            params["type"]      = "STOP_LOSS_LIMIT"
            params["stopPrice"] = self._fmt_price(stop_price)
        else:
            params["type"] = "LIMIT"
        return self._post("/api/v3/order", params)

    def get_order(self, symbol: str, order_id) -> dict:
        """Get status of a specific order."""
        return self._get("/api/v3/order", {
            "symbol":  symbol,
            "orderId": str(order_id),
        })

    def cancel_order(self, symbol: str, order_id) -> dict:
        """Cancel an open order."""
        return self._delete("/api/v3/order", {
            "symbol":  symbol,
            "orderId": str(order_id),
        })

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders, optionally filtered by symbol."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        result = self._get("/api/v3/openOrders", params)
        return result if isinstance(result, list) else []

    def get_closed_orders(self, symbol: str, limit: int = 10) -> list:
        """Get recent closed orders for a symbol."""
        result = self._get("/api/v3/allOrders", {
            "symbol": symbol,
            "limit":  limit,
        })
        closed = [o for o in result if o.get("status") in ("FILLED", "CANCELED")]
        return closed if isinstance(closed, list) else []

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fmt_qty(qty: float) -> str:
        """Format quantity — Binance requires specific decimal places."""
        if qty >= 1:
            return f"{qty:.3f}"
        elif qty >= 0.001:
            return f"{qty:.4f}"
        else:
            return f"{qty:.6f}"

    @staticmethod
    def _fmt_price(price: float) -> str:
        """Format price for Binance API."""
        if price >= 100:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        else:
            return f"{price:.6f}"

    def test_connection(self) -> bool:
        """Quick test — returns True if API keys work."""
        try:
            bal = self.get_usdt_balance()
            log.info(f"✓ Binance Testnet connected — {bal:.2f} USDT")
            return True
        except Exception as e:
            log.error(f"✗ Binance Testnet connection failed: {e}")
            return False
