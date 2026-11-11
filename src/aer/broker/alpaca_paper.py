"""Alpaca PAPER trading adapter, ruled 2026-09-24: "build it").

The simulator stays the decision engine; this adapter mirrors its target positions into an Alpaca
paper account so the 90-day record also exists at a third party, where it cannot be edited after
the fact. Plain REST over `requests` (already a dependency), so no new package enters uv.lock.

Safety: the base URL must be the paper host. A live URL raises at construction, and nothing in this
module can place a live order. Keys come only from the environment (APCA_API_KEY_ID,
APCA_API_SECRET_KEY, Alpaca's own names); this module never reads or writes.env.

Known limits of Alpaca paper fills (docs.alpaca.markets/us/docs/paper-trading): orders fill at the
NBBO with no slippage, market impact or dividends, and 10% of marketable orders partially fill at
random. The record reports that; it does not hide it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"
QTY_DECIMALS = 9  # Alpaca accepts up to nine decimal places for fractional qty
MIN_ORDER_QTY = 1e-6  # deltas below this are rounding noise, not a trade
MIN_NOTIONAL_USD = 1.0  # Alpaca's fractional minimum ("as little as $1"); smaller orders are rejected
HTTP_TIMEOUT_SEC = 20


class AlpacaNotConfigured(RuntimeError):
    """The paper keys are not in the environment."""


class AlpacaError(RuntimeError):
    """A non-2xx response from the Alpaca API."""


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float


class AlpacaPaper:
    def __init__(self, key_id: str, secret: str, base_url: str = PAPER_BASE_URL, session: Any = None) -> None:
        if "paper-api.alpaca.markets" not in base_url:
            raise ValueError(f"refusing a non-paper Alpaca endpoint: {base_url}")
        self._base = base_url.rstrip("/")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self._http = session or requests

    @classmethod
    def from_env(cls, session: Any = None) -> AlpacaPaper:
        key, secret = os.environ.get(KEY_ENV), os.environ.get(SECRET_ENV)
        if not key or not secret:
            raise AlpacaNotConfigured(f"{KEY_ENV} / {SECRET_ENV} not set")
        return cls(key, secret, session=session)

    def _call(self, method: str, path: str, body: dict | None = None) -> Any:
        r = self._http.request(
            method, f"{self._base}{path}", headers=self._headers, json=body, timeout=HTTP_TIMEOUT_SEC
        )
        if r.status_code >= 300:
            raise AlpacaError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else None

    def account(self) -> dict:
        return self._call("GET", "/v2/account")

    def clock(self) -> dict:
        return self._call("GET", "/v2/clock")

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._call("GET", "/v2/positions")}

    def position_prices(self) -> dict[str, float]:
        return {p["symbol"]: float(p.get("current_price") or 0.0) for p in self._call("GET", "/v2/positions")}

    def cancel_all_open(self) -> None:
        """DELETE /v2/orders cancels every open order, so a rerun never stacks a second copy."""
        self._call("DELETE", "/v2/orders")

    def order(self, order_id: str) -> dict:
        return self._call("GET", f"/v2/orders/{order_id}")

    def submit(
        self, symbol: str, side: str, qty: float, *, order_type: str = "market", limit_price: float | None = None
    ) -> dict:
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": "day",
            "qty": f"{round(qty, QTY_DECIMALS):.{QTY_DECIMALS}f}",
        }
        if limit_price is not None:
            body["limit_price"] = f"{limit_price:.2f}"
        return self._call("POST", "/v2/orders", body)

    def cancel(self, order_id: str) -> None:
        self._call("DELETE", f"/v2/orders/{order_id}")


def plan_orders(
    target: dict[str, float], held: dict[str, float], prices: dict[str, float] | None = None
) -> list[PlannedOrder]:
    """Orders that move `held` to `target` shares. Sells first, so cash is freed before buys.

    With `prices`, a delta worth less than MIN_NOTIONAL_USD is left as tolerated drift (Alpaca would
    reject it), except a sell that closes the whole position, which Alpaca accepts.
    """
    prices = prices or {}
    orders = []
    for sym in sorted(set(target) | set(held)):
        delta = round(target.get(sym, 0.0) - held.get(sym, 0.0), QTY_DECIMALS)
        if abs(delta) < MIN_ORDER_QTY:
            continue
        closes_position = delta < 0 and target.get(sym, 0.0) <= MIN_ORDER_QTY
        px = prices.get(sym)
        if px and abs(delta) * px < MIN_NOTIONAL_USD and not closes_position:
            continue
        orders.append(PlannedOrder(sym, "buy" if delta > 0 else "sell", abs(delta)))
    return sorted(orders, key=lambda o: (o.side != "sell", o.symbol))


def smoke(client: AlpacaPaper, spy_price: float | None = None) -> dict:
    """DoD Rung 2 "Alpaca paper connection smoke-tested with evidence": read account and clock, place a
    1-share SPY limit buy at half the last price (cannot fill, but inside any sane price band; $1.00
    when no price is known), read it back, cancel it, read the cancel back."""
    acct = client.account()
    clock = client.clock()
    limit = round(spy_price * 0.5, 2) if spy_price else 1.00
    placed = client.submit("SPY", "buy", 1, order_type="limit", limit_price=limit)
    client.cancel(placed["id"])
    after = client.order(placed["id"])
    return {
        "account_status": acct.get("status"),
        "account_currency": acct.get("currency"),
        "equity": acct.get("equity"),
        "clock_is_open": clock.get("is_open"),
        "order_id": placed["id"],
        "placed_status": placed.get("status"),
        "status_after_cancel": after.get("status"),
        "passed": acct.get("status") == "ACTIVE" and after.get("status") in ("canceled", "pending_cancel"),
    }


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    if "--smoke" in sys.argv:
        try:
            from aer.data.prices import get_current_price

            px = get_current_price("SPY")
        except Exception:
            px = None
        result = smoke(AlpacaPaper.from_env(), spy_price=px)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)
    print("usage: python -m aer.broker.alpaca_paper --smoke")
    sys.exit(2)
