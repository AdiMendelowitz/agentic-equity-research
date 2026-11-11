"""Price history data integrity validation."""

import pandas as pd

MIN_PRICE_ROWS: int = 504  # 2 trading years
MAX_DAILY_RETURN_PCT: float = 50.0


class PriceIntegrityError(ValueError):
    """Raised when a price DataFrame fails integrity checks."""


def validate_price_history(df: pd.DataFrame, ticker: str) -> None:
    """Validate raw OHLCV DataFrame before ML feature computation.

    Raises PriceIntegrityError on any of:
    1. Fewer than MIN_PRICE_ROWS rows
    2. More than 3 consecutive zero-volume rows
    3. Daily price jump > MAX_DAILY_RETURN_PCT with no split in that window
    4. Duplicate dates in index
    5. Any NaN in Close for the last 10 rows
    """
    if len(df) < MIN_PRICE_ROWS:
        raise PriceIntegrityError(f"{ticker}: only {len(df)} rows — need at least {MIN_PRICE_ROWS}")

    volume = df["Volume"]
    is_zero = (volume == 0).astype(int)
    consecutive = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).transform("sum")
    if (consecutive > 3).any():
        raise PriceIntegrityError(f"{ticker}: more than 3 consecutive zero-volume rows")

    close = df["Close"]
    returns = close.pct_change().abs() * 100
    big_jumps = returns > MAX_DAILY_RETURN_PCT
    if big_jumps.any():
        splits = df.get("Stock Splits", pd.Series(0.0, index=df.index))
        for idx in df.index[big_jumps]:
            loc = df.index.get_loc(idx)
            window_start = max(0, loc - 3)
            window_splits = splits.iloc[window_start : loc + 3]
            if window_splits.sum() == 0:
                raise PriceIntegrityError(
                    f"{ticker}: price jump >{MAX_DAILY_RETURN_PCT}% at {idx} without a recorded split"
                )

    if df.index.duplicated().any():
        raise PriceIntegrityError(f"{ticker}: duplicate dates in index")

    last_10_close = close.iloc[-10:]
    if last_10_close.isna().any():
        raise PriceIntegrityError(f"{ticker}: NaN values in Close for last 10 rows")
