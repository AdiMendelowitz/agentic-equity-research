"""Opinion tracking CRUD and analytics layer.

Single source of truth: SQLite opinions table.
Zero LLM calls in this module.
"""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from aer.data.cache import get_cache
from aer.models import HitRateMetrics, OpinionRecord
from aer.utils.opinion_constants import (
    MIN_OPINIONS_FOR_METRICS,
    MIN_VALID_RETURNS_FOR_SHARPE,
    SIGNAL_DIRECTION_MAP,
)

logger = logging.getLogger(__name__)

# R-D (Adi 2026-09-28), the writer admits only tickers the system watches or holds. tests/conftest.py
# switches this off for the suite, whose fixtures use synthetic tickers against a tmp database.
OPINION_TICKER_GUARD_ENABLED: bool = True
_HELD_STATE_PATH = Path(__file__).resolve().parents[2] / "simulator" / "portfolio_state.json"


def _load_watchlist() -> list[str]:
    from aer.utils.watchlist import load_watchlist

    return load_watchlist()


def _admissible_tickers() -> set[str] | None:
    """Watchlist plus held positions, upper-cased; None when the watchlist cannot be read.

    None means the guard stands aside: a guard fault must not destroy an opinion the pipeline produced, and a
    run whose watchlist cannot be read has already failed upstream (run_pipeline exits 2).
    """
    try:
        allowed = {t.upper() for t in _load_watchlist()}
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.error("record_opinion guard: watchlist unreadable (%s); guard not applied", exc)
        return None
    try:
        state = json.loads(_HELD_STATE_PATH.read_text(encoding="utf-8"))
        allowed |= {str(p.get("ticker", "")).upper() for p in state.get("positions", []) if p.get("ticker")}
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("record_opinion guard: held positions unreadable (%s); watchlist only", exc)
    return allowed


def ensure_admission_table(conn) -> None:
    """forward_return_admission: one row per ticker, how often its price fetch failed in a row and when.

    Also creates forward_return_terminal one row per opinion whose opinion_date has no price bar although
    bars exist on both sides of it. A weekend date is confirmed at once. A weekday gap (exchange holiday, or a hole
    in the vendor frame) is confirmed only when a later UTC calendar day sees the same gap, so one bad fetch cannot
    retire a row. Only confirmed rows leave the nightly scope; the opinions row is always kept. To put a row back in
    scope, DELETE its row from forward_return_terminal.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forward_return_admission (
            ticker TEXT PRIMARY KEY,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_attempt_utc TEXT,
            status TEXT CHECK(status IS NULL OR status = 'UNPRICEABLE')
        )
        """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forward_return_terminal (
            opinion_id INTEGER PRIMARY KEY,
            reason TEXT NOT NULL CHECK(reason = 'NO_BAR'),
            first_seen_utc TEXT NOT NULL,
            confirmed_utc TEXT
        )
        """)


def _note_no_bar(conn, opinion_id: int, opinion_date: date) -> bool:
    """Record a NO_BAR sighting; True when the row is now confirmed terminal, False while it is pending."""
    now = datetime.now(UTC)
    row = conn.execute(
        "SELECT first_seen_utc, confirmed_utc FROM forward_return_terminal WHERE opinion_id = ?", (opinion_id,)
    ).fetchone()
    if row is not None and row[1] is not None:
        return True
    if opinion_date.weekday() >= 5 or (row is not None and row[0][:10] < now.date().isoformat()):
        stamp = now.isoformat()
        conn.execute(
            "INSERT INTO forward_return_terminal(opinion_id, reason, first_seen_utc, confirmed_utc)"
            " VALUES (?, 'NO_BAR', ?, ?)"
            " ON CONFLICT(opinion_id) DO UPDATE SET confirmed_utc = excluded.confirmed_utc",
            (opinion_id, stamp, stamp),
        )
        return True
    if row is None:
        conn.execute(
            "INSERT INTO forward_return_terminal(opinion_id, reason, first_seen_utc) VALUES (?, 'NO_BAR', ?)",
            (opinion_id, now.isoformat()),
        )
    return False


def record_opinion(opinion: OpinionRecord) -> bool:
    """
    Insert opinion into SQLite opinions table.

    Args:
    opinion: OpinionRecord with all fields populated (forward_return_5d initially NULL)

    Returns:
    bool: True if inserted, False if UNIQUE constraint skipped (already exists)

    Validation
    - Reject if ticker is empty or > 5 chars
    - Reject if opinion_date > today
    - Reject if confidence or conviction is NaN or out of bounds

    Raises:
    sqlite3.Error: If database error (not UNIQUE constraint)

    NOTE: Does NOT raise on UNIQUE constraint failure; logs at DEBUG and returns False
    """
    import math
    import sqlite3
    import time

    # Validation: ticker
    if not opinion.ticker or len(opinion.ticker) > 5:
        logger.warning("record_opinion: rejected ticker %r (empty or > 5 chars)", opinion.ticker)
        return False

    # Validation: opinion_date
    if opinion.opinion_date > date.today():
        logger.warning("record_opinion: rejected opinion_date %s > today()", opinion.opinion_date)
        return False

    # Validation: confidence and conviction
    if math.isnan(opinion.confidence) or not (0.0 <= opinion.confidence <= 1.0):
        logger.warning("record_opinion: rejected invalid confidence %.2f", opinion.confidence)
        return False

    if math.isnan(opinion.conviction) or not (0.0 <= opinion.conviction <= 1.0):
        logger.warning("record_opinion: rejected invalid conviction %.2f", opinion.conviction)
        return False

    if OPINION_TICKER_GUARD_ENABLED:
        allowed = _admissible_tickers()
        if allowed is not None and opinion.ticker.upper() not in allowed:
            logger.warning("record_opinion: rejected %s, neither on the watchlist nor held", opinion.ticker)
            return False

    try:
        cache = get_cache()
        conn = cache._connect()

        # Prepare data
        opinion_date_str = opinion.opinion_date.isoformat()
        drivers_json = json.dumps(opinion.drivers) if opinion.drivers else json.dumps([])
        logged_at_ts = time.time()

        # Insert with parameterized query
        conn.execute(
            """
            INSERT INTO opinions
            (ticker, agent_name, opinion_date, signal, confidence, conviction, drivers, logged_at, forward_return_5d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opinion.ticker,
                opinion.agent_name,
                opinion_date_str,
                opinion.signal,
                opinion.confidence,
                opinion.conviction,
                drivers_json,
                logged_at_ts,
                opinion.forward_return_5d,
            ),
        )
        conn.commit()
        logger.info("record_opinion: inserted %s %s at %s", opinion.ticker, opinion.agent_name, opinion_date_str)
        return True

    except sqlite3.IntegrityError as e:
        # UNIQUE constraint violation - idempotent (already recorded)
        logger.debug("record_opinion: UNIQUE constraint for %s %s: %s", opinion.ticker, opinion.agent_name, e)
        return False

    except Exception as e:
        # Any other exception - don't break brief pipeline
        logger.warning(
            "record_opinion: exception inserting %s %s: %s",
            opinion.ticker,
            opinion.agent_name,
            e,
            exc_info=True,
        )
        return False


def get_opinions(
    ticker: str | None = None,
    agent_name: str | None = None,
    days: int | None = None,
) -> list[OpinionRecord]:
    """
    Retrieve opinions from SQLite, optionally filtered by ticker/agent/days.

    Args:
        ticker: Filter by ticker (e.g., "AAPL")
        agent_name: Filter by agent (e.g., "fundamental")
        days: If set, return only opinions from last N days

    Returns:
        List of OpinionRecord sorted by opinion_date DESC (newest first)

    Empty list if no matches.
    """
    import sqlite3
    import time

    try:
        cache = get_cache()
        conn = cache._connect()

        # Build WHERE clause dynamically
        where_clauses = []
        params = []

        if ticker:
            where_clauses.append("ticker = ?")
            params.append(ticker.upper())

        if agent_name:
            where_clauses.append("agent_name = ?")
            params.append(agent_name)

        if days is not None:
            cutoff_ts = time.time() - days * 86400
            where_clauses.append("logged_at >= ?")
            params.append(cutoff_ts)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = f"""
            SELECT ticker, agent_name, opinion_date, signal, confidence, conviction, drivers, logged_at, forward_return_5d
            FROM opinions
            WHERE {where_clause}
            ORDER BY opinion_date DESC
        """

        rows = conn.execute(query, params).fetchall()
        results = []

        for row in rows:
            (
                ticker_r,
                agent_name_r,
                opinion_date_str,
                signal,
                confidence,
                conviction,
                drivers_json,
                logged_at_ts,
                forward_return_5d,
            ) = row

            # Parse opinion_date from ISO string
            opinion_date_obj = datetime.fromisoformat(opinion_date_str).date()

            # Parse logged_at from Unix timestamp
            logged_at_dt = datetime.fromtimestamp(logged_at_ts, tz=UTC)

            # Parse drivers from JSON (handle NULL -> [])
            drivers = json.loads(drivers_json) if drivers_json else []

            record = OpinionRecord(
                ticker=ticker_r,
                agent_name=agent_name_r,
                opinion_date=opinion_date_obj,
                signal=signal,
                confidence=confidence,
                conviction=conviction,
                drivers=drivers,
                logged_at=logged_at_dt,
                forward_return_5d=forward_return_5d,
            )
            results.append(record)

        logger.info(
            "get_opinions: retrieved %d records (ticker=%s, agent=%s, days=%s)", len(results), ticker, agent_name, days
        )
        return results

    except sqlite3.OperationalError as e:
        logger.error("get_opinions: database unavailable: %s", e)
        return []

    except Exception as e:
        logger.error("get_opinions: exception: %s", e, exc_info=True)
        return []


def compute_hit_rates(ticker: str | None = None) -> list[HitRateMetrics]:
    """
    Compute per-agent per-ticker hit-rate metrics from recorded opinions.

    Filters by ticker if provided. Returns empty list if no opinions.

    Metrics: binary_hit_rate, return_weighted_score, sharpe_ratio, conviction_weighted_hit_rate.

    Filtering (minimum sample size for Sharpe):
    - Require >= MIN_OPINIONS_FOR_METRICS (20) total opinions per (agent, ticker)
    - Require >= MIN_VALID_RETURNS_FOR_SHARPE (20) non-NULL forward_return_5d for Sharpe
    - If either condition fails: all float metrics = None, insufficient_data = True

    Returns:
        List of HitRateMetrics, sorted by (agent_name, ticker)
    """
    from collections import defaultdict

    from aer.utils.opinion_constants import (
        BINARY_HIT_THRESHOLD,
    )

    opinions = get_opinions(ticker=ticker)
    if not opinions:
        return []

    groups: dict[tuple[str, str], list[OpinionRecord]] = defaultdict(list)
    for op in opinions:
        groups[(op.agent_name, op.ticker)].append(op)

    metrics_list = []
    for (agent_name, tick), agent_opinions in groups.items():
        valid_opinions = [op for op in agent_opinions if op.forward_return_5d is not None]
        n_total = len(agent_opinions)
        n_valid = len(valid_opinions)

        insufficient_data = n_total < MIN_OPINIONS_FOR_METRICS or n_valid < MIN_VALID_RETURNS_FOR_SHARPE

        if insufficient_data:
            metrics = HitRateMetrics(
                agent_name=agent_name,
                ticker=tick,
                total_opinions=n_total,
                binary_hit_rate=None,
                return_weighted_score=None,
                sharpe_ratio=None,
                conviction_weighted_hit_rate=None,
                n_profitable_holds=0,
                last_opinion_date=None,
                insufficient_data=True,
            )
        else:
            binary_hit_rate = _compute_binary_hit_rate(valid_opinions)
            return_weighted = _compute_return_weighted_score(valid_opinions)
            sharpe = _compute_sharpe_ratio(valid_opinions)
            conviction_weighted = _compute_conviction_weighted_hit_rate(valid_opinions)
            n_profitable = sum(1 for op in valid_opinions if op.forward_return_5d > BINARY_HIT_THRESHOLD)
            last_date = max(op.opinion_date for op in valid_opinions)

            metrics = HitRateMetrics(
                agent_name=agent_name,
                ticker=tick,
                total_opinions=n_total,
                binary_hit_rate=binary_hit_rate,
                return_weighted_score=return_weighted,
                sharpe_ratio=sharpe,
                conviction_weighted_hit_rate=conviction_weighted,
                n_profitable_holds=n_profitable,
                last_opinion_date=last_date,
                insufficient_data=False,
            )

        metrics_list.append(metrics)

    return sorted(metrics_list, key=lambda m: (m.agent_name, m.ticker))


def get_equity_curve(
    agent_name: str,
    ticker: str,
) -> dict[str, list[float]]:
    """
    Return cumulative returns for agent opinions on ticker.

    Args:
    agent_name: e.g., "fundamental"
    ticker: e.g., "AAPL"

    Returns:
    dict with keys:
    "dates": [isoformat dates], sorted oldest to newest
    "cumulative_returns": [cumulative values], starting at 1.0

    Empty dict {"dates": [], "cumulative_returns": []} if no opinions or all NULL returns.

    Filtering:
    - Exclude opinions with forward_return_5d = None
    - (non-overlapping windows, ruled 2026-09-23): an opinion is compounded only when its
    opinion_date is at least HOLDING_PERIOD_DAYS NYSE sessions after the previously compounded
    one, so each 5-day forward return covers a disjoint holding window. Opinions inside an
    open window are skipped, not averaged.
    - Compute cumulative = product(1 + forward_return_5d) over the compounded opinions only.
    """
    from aer.utils.opinion_constants import HOLDING_PERIOD_DAYS
    from aer.utils.trading_calendar import add_trading_days

    opinions = get_opinions(ticker=ticker, agent_name=agent_name)
    opinions = [op for op in opinions if op.forward_return_5d is not None]

    if not opinions:
        return {"dates": [], "cumulative_returns": []}

    opinions = sorted(opinions, key=lambda op: op.opinion_date)

    cumulative = 1.0
    dates = []
    cumulative_returns = []
    next_eligible = None

    for op in opinions:
        if next_eligible is not None and op.opinion_date < next_eligible:
            continue  # inside the previous opinion's holding window
        cumulative *= 1.0 + op.forward_return_5d
        dates.append(op.opinion_date.isoformat())
        cumulative_returns.append(cumulative)
        next_eligible = add_trading_days(op.opinion_date, HOLDING_PERIOD_DAYS)

    return {
        "dates": dates,
        "cumulative_returns": cumulative_returns,
    }


def populate_forward_returns() -> int:
    """
    Fetch 5-day forward returns for opinions without forward_return_5d populated.

    Called by nightly_metrics.py after market close.

    Returns:
    int: count of opinions updated

    Edge case handling (all explicit):
    1. Missing price data on forward_date: skip opinion, leave forward_return_5d = None, log WARNING
    2. Missing price data on opinion_date: skip opinion, leave forward_return_5d = None. When bars exist on both
    sides of the date (weekend, exchange holiday) the row is recorded in forward_return_terminal as NO_BAR confirmed at once on a weekend, on the second UTC day that sees the gap otherwise, and only then
    dropped from scope. A date outside the fetched window still logs a WARNING and stays in scope.
    3. Corporate actions: ADJUSTED basis(B), get_price_history -> auto_adjust=True).
    Dividends ARE folded in, so this is a total return over the holding window, not a price
    return. Splits are adjusted on both yfinance paths (prices.py:118-120), so no split
    guard is needed here. This DIFFERS from ml/features.py's 'label', which is a
    cost-thresholded binary on the dividend-UNADJUSTED path. Same nominal quantity, two
    definitions, both intentional;
    4. Timezone: opinion_date treated as end-of-day (16:00 ET) close
    5. Market holidays: no calendar library is consulted. The forward bar is the
    HOLDING_PERIOD_DAYS-th bar after the opinion bar in the fetched frame, and a bar exists
    only on a trading day. Weekends and every exchange holiday are handled by
    construction. with OPINION_TICKER_GUARD_ENABLED, tickers outside the watchlist and held set are not fetched, unless
    their newest NULL row is within FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS. Their rows are kept.

    Assumes: aer.data.prices.get_price_history available (ADJUSTED path,(B)). Batch updates wrapped in explicit transaction boundaries.
    """
    from datetime import date as date_type

    import pandas as pd

    from aer.data.prices import get_price_history
    from aer.utils import opinion_constants as oc
    from aer.utils.opinion_constants import (
        FORWARD_RETURN_MAX_FAILED_ATTEMPTS,
        FORWARD_RETURN_PERIOD_LADDER_BARS,
        HOLDING_PERIOD_DAYS,
    )

    MAX_FORWARD_RETURN_TICKERS_PER_RUN = oc.MAX_FORWARD_RETURN_TICKERS_PER_RUN  # read at call time (tests patch it)

    # NOTE get_price_history is imported HERE, inside the function, deliberately.
    # tests/test_opinion_tracking.py::TestPopulateForwardReturns monkeypatches
    # aer.data.prices.get_price_history. Hoisting this import to module scope binds the real
    # function at import time and sends all five of those tests to the live network without any
    # of them failing. Do not hoist it.

    cache = get_cache()
    conn = cache._connect()
    ensure_admission_table(conn)
    conn.commit()

    rows = conn.execute(
        "SELECT id, ticker, opinion_date FROM opinions WHERE forward_return_5d IS NULL"
        " AND id NOT IN (SELECT opinion_id FROM forward_return_terminal WHERE confirmed_utc IS NOT NULL)"
        " ORDER BY opinion_date ASC"
    ).fetchall()

    logger.info("populate_forward_returns: %d opinions to process", len(rows))
    if not rows:
        logger.info("populate_forward_returns complete: 0 updated, 0 skipped, 0 unclassified (no NULL rows)")
        return 0

    # Group by ticker: one price fetch per ticker per run, not one per row. `rows` is ordered oldest-first.
    by_ticker: dict[str, list[tuple]] = {}
    for opinion_id, ticker, opinion_date_str in rows:
        by_ticker.setdefault(ticker, []).append((opinion_id, opinion_date_str))

    # (Adi 2026-09-29): admit only the tickers record_opinion admits, the watchlist plus held positions. SPY,
    # TEST and five TICK tickers reached the fetch because the write guard postdates their rows. Their rows stay.
    if OPINION_TICKER_GUARD_ENABLED:
        allowed = _admissible_tickers()
        if allowed is not None:
            cutoff = date_type.fromordinal(date_type.today().toordinal() - oc.FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS)
            outside = sorted(
                t
                for t in by_ticker
                if t.upper() not in allowed and date_type.fromisoformat(by_ticker[t][-1][1]) < cutoff
            )
            for t in outside:
                del by_ticker[t]
            if outside:
                logger.info(
                    "populate_forward_returns: %d tickers outside the watchlist and held set (newest NULL row older "
                    "than %d days) not admitted: %s",
                    len(outside),
                    oc.FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS,
                    ", ".join(outside),
                )

    # R-D, oldest-first admission let 117 unpriceable tickers hold every slot forever. Admission is now
    # never-attempted first (oldest backlog first among them), then least recently attempted; UNPRICEABLE
    # tickers are never admitted.
    admission = {
        t: (attempts, last, status)
        for t, attempts, last, status in conn.execute(
            "SELECT ticker, attempts, last_attempt_utc, status FROM forward_return_admission"
        )
    }
    retired = [t for t in by_ticker if admission.get(t, (0, None, None))[2] == "UNPRICEABLE"]
    for t in retired:
        del by_ticker[t]
    if retired:
        logger.info("populate_forward_returns: %d UNPRICEABLE tickers not admitted", len(retired))
    if not by_ticker:
        logger.info(
            "populate_forward_returns complete: 0 updated "
            "(every NULL row belongs to an UNPRICEABLE or non-admitted ticker)"
        )
        return 0
    order = sorted(
        by_ticker,
        key=lambda t: (
            admission.get(t, (0, None, None))[1] is not None,
            admission.get(t, (0, None, None))[1] or "",
            by_ticker[t][0][1],
        ),
    )

    def _note_attempt(ticker: str, fetched: bool) -> None:
        now = datetime.now(UTC).isoformat()
        if fetched:
            conn.execute(
                "INSERT INTO forward_return_admission(ticker, attempts, last_attempt_utc, status)"
                " VALUES (?, 0, ?, NULL) ON CONFLICT(ticker) DO UPDATE SET attempts = 0,"
                " last_attempt_utc = excluded.last_attempt_utc, status = NULL",
                (ticker, now),
            )
            return
        attempts = admission.get(ticker, (0, None, None))[0] + 1
        status = "UNPRICEABLE" if attempts >= FORWARD_RETURN_MAX_FAILED_ATTEMPTS else None
        conn.execute(
            "INSERT INTO forward_return_admission(ticker, attempts, last_attempt_utc, status) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(ticker) DO UPDATE SET attempts = excluded.attempts,"
            " last_attempt_utc = excluded.last_attempt_utc, status = excluded.status",
            (ticker, attempts, now, status),
        )
        if status:
            logger.warning(
                "populate_forward_returns: %s marked UNPRICEABLE after %d failed fetches in a row", ticker, attempts
            )

    admitted = order[:MAX_FORWARD_RETURN_TICKERS_PER_RUN]
    in_scope = sum(len(by_ticker[name]) for name in admitted)
    if len(by_ticker) > MAX_FORWARD_RETURN_TICKERS_PER_RUN:
        logger.warning(
            "populate_forward_returns: %d tickers carry NULL rows, exceeding the per-run cap of %d; "
            "admitting %d, never-attempted first then least recently attempted (%d of %d rows), rest deferred",
            len(by_ticker),
            MAX_FORWARD_RETURN_TICKERS_PER_RUN,
            len(admitted),
            in_scope,
            sum(len(v) for v in by_ticker.values()),
        )

    # One quantized period for the whole run. Calendar-day arithmetic is used ONLY for
    # fetch-window sizing, which permits explicitly; it never touches the return.
    oldest_date = min(date_type.fromisoformat(by_ticker[t][0][1]) for t in admitted)
    span_days = (date_type.today() - oldest_date).days
    period_bars = next((bars for bars in FORWARD_RETURN_PERIOD_LADDER_BARS if bars >= span_days), None)
    if period_bars is None:
        period_bars = FORWARD_RETURN_PERIOD_LADDER_BARS[-1]
        logger.warning(
            "populate_forward_returns: backlog span %d calendar days exceeds the largest ladder rung (%d bars); "
            "opinions older than the fetched window will skip as out of range until the ladder is extended",
            span_days,
            period_bars,
        )
    period = f"{period_bars}d"
    logger.info(
        "populate_forward_returns: %d tickers admitted, %d rows in scope, period=%s (oldest NULL row overall %s, span %d calendar days)",
        len(admitted),
        in_scope,
        period,
        oldest_date,
        span_days,
    )

    updated = 0
    skipped = 0
    unclassified = 0
    no_bar_confirmed = 0
    no_bar_pending = 0

    # Wrap batch updates in explicit transaction
    try:
        conn.execute("BEGIN TRANSACTION")

        for ticker in admitted:
            ticker_rows = by_ticker[ticker]

            try:
                history = get_price_history(ticker, period=period, interval="1d")
            except (OSError, ValueError, KeyError) as e:
                logger.error("Price fetch error for %s (period=%s): %s", ticker, period, e)
                skipped += len(ticker_rows)
                _note_attempt(ticker, fetched=False)
                continue
            except Exception:
                # Breadth is deliberate: one ticker must not abort the backfill. It is COUNTED
                # rather than forgiven, so an unclassified failure is visible in the summary.
                logger.exception("Unclassified price fetch failure for %s (period=%s); continuing", ticker, period)
                skipped += len(ticker_rows)
                unclassified += len(ticker_rows)
                _note_attempt(ticker, fetched=False)
                continue

            if history.empty:
                logger.warning("No price history for %s over period=%s", ticker, period)
                skipped += len(ticker_rows)
                _note_attempt(ticker, fetched=False)
                continue
            _note_attempt(ticker, fetched=True)

            # -916): the price index IS the trading calendar of record, since
            # a bar exists only on a trading day. The forward date is the HOLDING_PERIOD_DAYS-th
            # bar after the opinion bar, derived positionally. No add_trading_days, no BDay, no
            # holiday table: all three are the class, and all three can disagree with the
            # very price series being indexed.
            try:
                bar_dates = list(history.index.date)
                position_of = {bar_date: i for i, bar_date in enumerate(bar_dates)}
                closes = history["Close"]
            except (AttributeError, KeyError):
                # A non-datetime index or an absent Close column must not escape to the outer
                # transaction handler, which would roll back EVERY ticker's updates and abort the
                # nightly run over one malformed frame.
                logger.exception("Unusable price frame for %s (period=%s); continuing", ticker, period)
                skipped += len(ticker_rows)
                unclassified += len(ticker_rows)
                continue

            for opinion_id, opinion_date_str in ticker_rows:
                try:
                    opinion_date = date_type.fromisoformat(opinion_date_str)

                    i = position_of.get(opinion_date)
                    if i is None and bar_dates[0] < opinion_date < bar_dates[-1]:
                        # bars exist on both sides, so the price index itself says this is not a trading day.
                        if _note_no_bar(conn, opinion_id, opinion_date):
                            no_bar_confirmed += 1
                        else:
                            no_bar_pending += 1
                        skipped += 1
                        continue
                    if i is None:
                        logger.warning(
                            "No price bar for %s on opinion_date %s (outside the fetched window, or not a trading day)",
                            ticker,
                            opinion_date,
                        )
                        skipped += 1
                        continue

                    j = i + HOLDING_PERIOD_DAYS
                    if j >= len(bar_dates):
                        logger.warning(
                            "Insufficient forward window for %s from %s: needs bar %d, frame holds %d",
                            ticker,
                            opinion_date,
                            j,
                            len(bar_dates),
                        )
                        skipped += 1
                        continue

                    forward_date = bar_dates[j]

                    opinion_price = closes.iloc[i]
                    if pd.isna(opinion_price) or opinion_price <= 0:
                        logger.warning("Invalid price for %s on %s: %s", ticker, opinion_date, opinion_price)
                        skipped += 1
                        continue

                    forward_price = closes.iloc[j]
                    if pd.isna(forward_price) or forward_price <= 0:
                        logger.warning("Invalid price for %s on %s: %s", ticker, forward_date, forward_price)
                        skipped += 1
                        continue

                    forward_return = (forward_price - opinion_price) / opinion_price

                    conn.execute(
                        "UPDATE opinions SET forward_return_5d = ? WHERE id = ?",
                        (float(forward_return), opinion_id),
                    )
                    updated += 1
                    logger.debug(
                        "Updated opinion %d: %s %s -> %s forward_return=%.4f",
                        opinion_id,
                        ticker,
                        opinion_date,
                        forward_date,
                        forward_return,
                    )

                except Exception as e:
                    logger.error(
                        "Error computing forward return for %s on %s (id=%d): %s",
                        ticker,
                        opinion_date_str,
                        opinion_id,
                        e,
                        exc_info=True,
                    )
                    skipped += 1
                    unclassified += 1

        # Commit only after all updates succeed
        conn.commit()
        logger.info(
            "populate_forward_returns complete: %d updated, %d skipped, %d unclassified (of %d rows in scope)",
            updated,
            skipped,
            unclassified,
            in_scope,
        )
        if no_bar_confirmed or no_bar_pending:
            logger.info(
                "populate_forward_returns: NO_BAR rows (no price bar on their opinion_date): %d confirmed and out of "
                "scope from the next run, %d pending a second sighting on a later day",
                no_bar_confirmed,
                no_bar_pending,
            )
        if updated == 0 and not (no_bar_confirmed or no_bar_pending):
            #'s signature was updated=0 read as a no-op for months. It is a WARNING now.
            logger.warning(
                "populate_forward_returns: %d rows in scope across %d admitted tickers and 0 updated - "
                "the backfill wrote nothing this run",
                in_scope,
                len(admitted),
            )

    except Exception as e:
        logger.error("Batch transaction failed - rolling back all updates: %s", e, exc_info=True)
        try:
            conn.rollback()
        except Exception as rb_err:
            logger.error("Rollback failed: %s", rb_err)
        raise

    return updated


def _compute_binary_hit_rate(opinions: list[OpinionRecord]) -> float:
    """
    Compute % of opinions with correct signal direction.

    HOLD uses |return| < HOLD_THRESHOLD (1%) for correctness.
    AVOID treated same as SELL (signal_dir = -1).
    """
    from aer.utils.opinion_constants import BINARY_HIT_THRESHOLD, HOLD_THRESHOLD

    if not opinions:
        return 0.0

    correct = 0
    for op in opinions:
        if op.forward_return_5d is None:
            continue

        signal = op.signal
        forward_return = op.forward_return_5d

        if signal == "HOLD":
            correct_pred = abs(forward_return) < HOLD_THRESHOLD
        else:
            signal_dir = SIGNAL_DIRECTION_MAP.get(signal, 0.0)
            correct_pred = (signal_dir * forward_return) > BINARY_HIT_THRESHOLD

        if correct_pred:
            correct += 1

    return (correct / len(opinions)) * 100.0 if opinions else 0.0


def _compute_return_weighted_score(opinions: list[OpinionRecord]) -> float:
    """
    Compute mean of (signal_direction * forward_return).

    Formally defined
    return_weighted_score = mean([r_i * sign(signal_i) for r_i in forward_returns if r_i is not NULL])
    where sign(BUY)=1, sign(SELL)=-1, sign(HOLD)=0, sign(AVOID)=-1
    """

    if not opinions:
        return 0.0

    score = 0.0
    for op in opinions:
        if op.forward_return_5d is None:
            continue

        signal_dir = SIGNAL_DIRECTION_MAP.get(op.signal, 0.0)
        score += signal_dir * op.forward_return_5d

    return score / len([op for op in opinions if op.forward_return_5d is not None])


def _compute_sharpe_ratio(opinions: list[OpinionRecord]) -> float:
    """
    Compute annualized Sharpe ratio from 5-day returns.

    Requires >= 20 samples; minimum is enforced by caller (compute_hit_rates).
    Formula: Sharpe = (mean_return / std_return) * sqrt(252)
    where 252 = trading days/year, and individual 5-day returns treated as independent periods.
    """
    import numpy as np

    if len(opinions) < 2:
        return 0.0

    returns = np.array([op.forward_return_5d for op in opinions if op.forward_return_5d is not None])
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0.0 or np.isnan(std_return):
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(252)
    return float(sharpe)


def _compute_conviction_weighted_hit_rate(opinions: list[OpinionRecord]) -> float:
    """
    Compute hit-rate weighted by conviction. Emphasizes high-conviction opinions.
    conviction_weighted_hit_rate = sum(binary_hit_i * conviction_i) / sum(conviction_i)
    """
    from aer.utils.opinion_constants import BINARY_HIT_THRESHOLD, HOLD_THRESHOLD

    if not opinions:
        return 0.0

    weighted_hits = 0.0
    total_conviction = 0.0

    for op in opinions:
        if op.forward_return_5d is None:
            continue

        signal = op.signal
        forward_return = op.forward_return_5d

        if signal == "HOLD":
            binary_hit = 1.0 if abs(forward_return) < HOLD_THRESHOLD else 0.0
        else:
            signal_dir = SIGNAL_DIRECTION_MAP.get(signal, 0.0)
            binary_hit = 1.0 if (signal_dir * forward_return) > BINARY_HIT_THRESHOLD else 0.0

        conviction = op.conviction if op.conviction is not None else 0.5
        weighted_hits += binary_hit * conviction
        total_conviction += conviction

    return (weighted_hits / total_conviction * 100.0) if total_conviction > 0 else 0.0
