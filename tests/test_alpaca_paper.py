"""(ruled 2026-09-24, "build it"): the Alpaca PAPER mirror. No network: a fake session stands in
for the REST API. The first test is the safety invariant: no live endpoint, ever."""

from __future__ import annotations

import json

import pytest

from aer.broker.alpaca_paper import AlpacaNotConfigured, AlpacaPaper, PlannedOrder, plan_orders, smoke


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        return self._body


class FakeAlpaca:
    """Enough of the v2 API: account, clock, positions, orders (create, get, delete)."""

    def __init__(self, is_open=True, positions=None, fill=True):
        self.is_open, self.pos, self.fill, self.orders, self.calls = is_open, dict(positions or {}), fill, {}, []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append((method, url, json))
        assert headers["APCA-API-KEY-ID"] == "k" and headers["APCA-API-SECRET-KEY"] == "s"
        path = url.split("paper-api.alpaca.markets", 1)[1]
        if path == "/v2/account":
            return _Resp(200, {"status": "ACTIVE", "currency": "USD", "equity": "100000"})
        if path == "/v2/clock":
            return _Resp(200, {"is_open": self.is_open, "next_open": "2026-09-25T09:30:00-04:00"})
        if path == "/v2/positions":
            return _Resp(
                200,
                [{"symbol": s, "qty": str(q), "current_price": "100.0"} for s, q in self.pos.items() if abs(q) > 1e-9],
            )
        if path == "/v2/orders" and method == "DELETE":
            for o in self.orders.values():
                if o["status"] in ("new", "partially_filled"):
                    o["status"] = "canceled"
            return _Resp(207, [])
        if path == "/v2/orders" and method == "POST":
            if json["symbol"] in getattr(self, "reject", ()):
                return _Resp(422, {"message": "asset is not fractionable"})
            oid = f"o{len(self.orders)}"
            qty = float(json["qty"])
            o = {"id": oid, "status": "new", "symbol": json["symbol"], "side": json["side"], "qty": json["qty"]}
            if self.fill and json["type"] == "market":
                self.pos[json["symbol"]] = self.pos.get(json["symbol"], 0.0) + (qty if json["side"] == "buy" else -qty)
                o.update(status="filled", filled_qty=json["qty"], filled_avg_price="100.00")
            self.orders[oid] = o
            return _Resp(200, o)
        if path.startswith("/v2/orders/") and method == "DELETE":
            self.orders[path.rsplit("/", 1)[1]]["status"] = "canceled"
            return _Resp(204, None)
        if path.startswith("/v2/orders/"):
            return _Resp(200, self.orders[path.rsplit("/", 1)[1]])
        return _Resp(404, {"message": "not found"})


def test_live_endpoint_is_refused():
    with pytest.raises(ValueError, match="non-paper"):
        AlpacaPaper("k", "s", base_url="https://api.alpaca.markets")


def test_missing_keys_raise_not_configured(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaNotConfigured):
        AlpacaPaper.from_env()


def test_plan_orders_sells_first_and_ignores_rounding_noise():
    plan = plan_orders({"AAA": 2.5, "BBB": 1.0, "CCC": 1.0}, {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0000000001, "DDD": 3.0})
    assert plan == [PlannedOrder("DDD", "sell", 3.0), PlannedOrder("AAA", "buy", 1.5)]


def test_smoke_places_an_unfillable_limit_and_cancels_it():
    fake = FakeAlpaca()
    out = smoke(AlpacaPaper("k", "s", session=fake))
    assert out["passed"] is True and out["status_after_cancel"] == "canceled"
    post = [c for c in fake.calls if c[0] == "POST"][0][2]
    assert post["type"] == "limit" and post["limit_price"] == "1.00" and post["symbol"] == "SPY"
    fake2 = FakeAlpaca()
    smoke(AlpacaPaper("k", "s", session=fake2), spy_price=660.0)
    assert [c for c in fake2.calls if c[0] == "POST"][0][2]["limit_price"] == "330.00"


@pytest.fixture
def mirror(monkeypatch, tmp_path):
    import aer.operations.alpaca_mirror as m

    state = tmp_path / "portfolio_state.json"
    state.write_text(
        json.dumps({"positions": [{"ticker": "NVDA", "shares": 1.25}, {"ticker": "BRK-B", "shares": 0.5}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(m, "_STATE_FILE", state)
    monkeypatch.setattr(m, "_MIRROR_LOG", tmp_path / "alpaca_mirror_log.csv")
    return m


def _status():
    import os
    from pathlib import Path

    return json.loads((Path(os.environ["AER_ARTIFACT_DIR"]) / "alpaca_mirror_status.json").read_text(encoding="utf-8"))


def test_mirror_reconciles_the_paper_account_to_the_simulator(mirror):
    fake = FakeAlpaca(positions={"NVDA": 0.25, "OLD": 2.0})
    assert mirror.run(AlpacaPaper("k", "s", session=fake), sleep=lambda s: None) == 0
    assert fake.pos["NVDA"] == pytest.approx(1.25) and fake.pos["BRK.B"] == pytest.approx(0.5)
    assert fake.pos["OLD"] == pytest.approx(0.0)
    st = _status()
    assert st["exit_code"] == 0 and st["orders"] == 3 and st["drift"] == {}
    assert (mirror._MIRROR_LOG).read_text(encoding="utf-8").count("\n") == 4  # header + 3 orders


def test_mirror_does_nothing_while_the_market_is_closed(mirror):
    fake = FakeAlpaca(is_open=False)
    assert mirror.run(AlpacaPaper("k", "s", session=fake), sleep=lambda s: None) == 0
    assert not [c for c in fake.calls if c[0] == "POST"]
    assert _status()["skipped"] == "market closed"


def test_unfilled_orders_leave_drift_and_exit_1(mirror):
    fake = FakeAlpaca(fill=False)
    assert mirror.run(AlpacaPaper("k", "s", session=fake), sleep=lambda s: None) == 1
    assert set(_status()["drift"]) == {"NVDA", "BRK.B"}


def test_missing_keys_are_a_fatal_status(mirror, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    assert mirror.main() == 2
    assert "AlpacaNotConfigured" in _status()["error"]


def test_orders_below_the_one_dollar_minimum_are_left_as_tolerated_drift():
    plan = plan_orders({"AAA": 1.005, "BBB": 0.0}, {"AAA": 1.0, "BBB": 0.001}, {"AAA": 60.0, "BBB": 60.0})
    assert plan == [PlannedOrder("BBB", "sell", 0.001)]  # closing a position is allowed below $1


def test_one_rejected_order_does_not_stop_the_rest_and_everything_is_logged(mirror):
    fake = FakeAlpaca()
    fake.reject = ("NVDA",)
    assert mirror.run(AlpacaPaper("k", "s", session=fake), sleep=lambda s: None) == 1
    assert fake.pos.get("BRK.B") == pytest.approx(0.5)
    st = _status()
    assert st["rejected"][0]["symbol"] == "NVDA"
    log = mirror._MIRROR_LOG.read_text(encoding="utf-8")
    assert "rejected" in log and "BRK.B" in log


def test_open_orders_are_cancelled_before_planning_and_after_the_wait(mirror):
    fake = FakeAlpaca(fill=False)
    mirror.run(AlpacaPaper("k", "s", session=fake), sleep=lambda s: None)
    deletes = [c for c in fake.calls if c[0] == "DELETE" and c[1].endswith("/v2/orders")]
    assert len(deletes) == 2 and all(o["status"] == "canceled" for o in fake.orders.values())
