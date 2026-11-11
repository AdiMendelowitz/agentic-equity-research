"""Ken French factor data wrapper with SQLite caching.

Fetches Fama-French 5-factor daily data directly from the Ken French Data Library
(HTTP + ZIP, no pandas_datareader dependency).
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import date

import pandas as pd
import requests

from aer.utils.cache import TTL_PRICES, data_cache
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)

_FF5_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"
_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
_CACHE_KEY_TEMPLATE = "factors:ken_french_daily:{date_str}"


def _df_to_cache(df: pd.DataFrame) -> dict:
    return {
        "columns": list(df.columns),
        "index": [str(v) for v in df.index],
        "data": df.values.tolist(),
    }


def _df_from_cache(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        payload["data"],
        index=pd.to_datetime(payload["index"]).date,
        columns=payload["columns"],
    )


def _fetch_ken_french_ff5(end_date: date | None = None) -> pd.DataFrame:
    """Download FF5 daily factors from the Ken French Data Library (HTTP + ZIP).

    Returns a DataFrame indexed by datetime.date with columns Mkt-RF, SMB, HML, RMW, CMA, RF.
    Values are in decimal (Ken French publishes percentages; divided by 100 here).
    """
    resp = requests.get(_FF5_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        content = zf.read(csv_name).decode("latin-1")

    # Extract only the numeric data lines: 8-digit date followed by comma and values.
    # Ken French CSV format: YYYYMMDD,val1,val2,... (skip header lines and comments)
    data_lines = [line for line in content.splitlines() if re.match(r"^\d{8},", line)]

    if not data_lines:
        logger.warning("_fetch_ken_french_ff5: no data rows parsed from ZIP contents")
        return pd.DataFrame(columns=_FF5_COLUMNS)

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=",",
        header=None,
        names=["date"] + _FF5_COLUMNS,
        dtype={"date": str},
        engine="python",
    )
    df.index = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
    df = df.drop("date", axis=1)
    df = df / 100.0  # Ken French publishes percentages → decimal

    if end_date is not None:
        df = df[df.index <= end_date]

    return df


def get_factor_data(as_of: date | None = None) -> pd.DataFrame:
    """Fetch Fama-French 5-factor daily data from the Ken French Data Library.

    On cache miss: calls _fetch_ken_french_ff5 once and stores in data_cache().
    On cache hit: returns directly from cache without a network call.

    Args:
        as_of: Optional point-in-time cutoff date. Rows after this date are excluded.
            Defaults to None (uses today's date for the cache key).

    Returns:
        DataFrame indexed by datetime.date with columns Mkt-RF, SMB, HML, RMW, CMA, RF.
        Values are in decimal (e.g. 0.005 means 0.5%).
    """
    effective_date = as_of or date.today()
    date_str = effective_date.isoformat()
    cache_key = _CACHE_KEY_TEMPLATE.format(date_str=date_str)

    cache = data_cache()
    cached = cache.get(cache_key)
    if cached is not None and isinstance(cached, dict) and "data" in cached:
        df = _df_from_cache(cached)
        logger.debug("factors cache hit: %s (%d rows)", cache_key, len(df))
        record_vintage("factors")
        return df

    logger.info("factors cache miss — fetching %s from Ken French Data Library", _FF5_DATASET)
    df = _fetch_ken_french_ff5(end_date=effective_date)

    if df.empty:
        logger.warning("factors: empty result — skipping cache for %s", cache_key)
        record_vintage("factors")
        return df

    cache.set(cache_key, _df_to_cache(df), ttl=TTL_PRICES)
    logger.info("factors cached: %s (%d rows)", cache_key, len(df))
    record_vintage("factors")
    return df
