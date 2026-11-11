"""Mirror the simulator's positions into the Alpaca paper account.

Reads simulator/portfolio_state.json (the simulator stays the decision engine), plans the share
deltas against the paper account, submits them as fractional market DAY orders during regular
hours only, waits briefly for fills, and records every order plus the reconciled positions:
 simulator/alpaca_mirror_log.csv one row per order (appended)
 artifacts/briefs/alpaca_mirror_status.json heartbeat + exit code for the audit and the logon check
Exit 0: in sync (or market closed, nothing to do today); 1: drift remains after the fill wait;
2: fatal (keys missing, API error). Holdings in the paper account that the simulator does not hold
are sold, so the account tracks the simulator exactly.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from aer.broker.alpaca_paper import MIN_NOTIONAL_USD, AlpacaPaper, plan_orders  # noqa: E402
from aer.paths import ENV_FILE
from aer.utils.artifact_schema import artifact_dir  # noqa: E402

logger = logging.getLogger(__name__)
_STATE_FILE = Path(__file__).parent / "portfolio_state.json"
_MIRROR_LOG = Path(__file__).parent / "alpaca_mirror_log.csv"
STATUS_FILENAME = "alpaca_mirror_status.json"
FILL_WAIT_SEC = 60
DRIFT_TOLERANCE_QTY = 1e-4
_FIELDS = ["utc", "symbol", "side", "qty", "order_id", "status", "filled_qty", "filled_avg_price"]


def _write_status(exit_code: int, **fields) -> None:
    try:
        path = artifact_dir() / STATUS_FILENAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"generated_at_utc": datetime.now(UTC).isoformat(), "exit_code": exit_code, **fields},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception as e:
        logger.error("could not write %s: %s", STATUS_FILENAME, e)


def to_alpaca_symbol(ticker: str) -> str:
    """yfinance writes share classes with '-' (BRK-B); Alpaca uses '.' (BRK.B)."""
    return ticker.upper().replace("-", ".")


def _target() -> dict[str, float]:
    state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for p in state.get("positions", []):
        sym = to_alpaca_symbol(p["ticker"])
        out[sym] = out.get(sym, 0.0) + float(p["shares"])
    return out


def _target_prices() -> dict[str, float]:
    """Fallback price per symbol for the $1 minimum check: the simulator's entry price."""
    state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    return {to_alpaca_symbol(p["ticker"]): float(p.get("entry_price") or 0.0) for p in state.get("positions", [])}


def run(client: AlpacaPaper, *, sleep=time.sleep) -> int:
    clock = client.clock()
    if not clock.get("is_open"):
        _write_status(0, skipped="market closed", next_open=clock.get("next_open"))
        return 0
    client.cancel_all_open()  # a rerun must never stack a second copy of an unfilled order
    target = _target()
    prices = {**_target_prices(), **{k: v for k, v in client.position_prices().items() if v}}
    orders = plan_orders(target, client.positions(), prices)
    placed, failed = [], []
    rows = []
    now = datetime.now(UTC).isoformat()
    try:
        for o in orders:
            try:
                placed.append((o, client.submit(o.symbol, o.side, o.qty)))
            except Exception as e:  # one rejected order (non-fractionable, below minimum) must not stop the rest
                failed.append({"symbol": o.symbol, "side": o.side, "qty": o.qty, "error": str(e)[:300]})
                rows.append(
                    {
                        "utc": now,
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": o.qty,
                        "order_id": "",
                        "status": "rejected: " + str(e)[:120],
                        "filled_qty": 0,
                        "filled_avg_price": "",
                    }
                )
        final = {}
        waited = 0
        while placed and waited <= FILL_WAIT_SEC:
            final = {r["id"]: client.order(r["id"]) for _, r in placed}
            if all(f.get("status") in ("filled", "canceled", "rejected", "expired") for f in final.values()):
                break
            sleep(5)
            waited += 5
        client.cancel_all_open()  # leftovers are re-planned tomorrow from the positions, never stacked
        for o, r in placed:
            f = final.get(r["id"], r)
            rows.append(
                {
                    "utc": now,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": o.qty,
                    "order_id": r["id"],
                    "status": f.get("status"),
                    "filled_qty": f.get("filled_qty"),
                    "filled_avg_price": f.get("filled_avg_price"),
                }
            )
    finally:
        new = not _MIRROR_LOG.exists()
        with _MIRROR_LOG.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    held = client.positions()
    drift = {}
    for sym in set(target) | set(held):
        d = target.get(sym, 0.0) - held.get(sym, 0.0)
        px = prices.get(sym, 0.0)
        if abs(d) > DRIFT_TOLERANCE_QTY and (not px or abs(d) * px >= MIN_NOTIONAL_USD):  # unknown price counts
            drift[sym] = round(d, 6)
    code = 0 if not drift and not failed else 1
    _write_status(code, orders=len(placed), rejected=failed, drift=drift, positions=held)
    return code


def main() -> int:
    try:
        return run(AlpacaPaper.from_env())
    except Exception as e:  # keys missing, API down: visible, never silent
        logger.error("alpaca mirror fatal: %s", e, exc_info=True)
        _write_status(2, error=f"{type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    from dotenv import load_dotenv  # keys live in .env; loaded only when run as the scheduled job

    load_dotenv(ENV_FILE, override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
