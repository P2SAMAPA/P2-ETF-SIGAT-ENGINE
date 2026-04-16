"""
US market calendar utilities.
"""
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import pandas as pd


def get_us_calendar():
    """Return NYSE calendar."""
    return mcal.get_calendar("NYSE")


def next_trading_day(date: datetime = None) -> datetime:
    """
    Return the next trading day after given date.
    If date is None, use current UTC time.
    """
    if date is None:
        date = datetime.utcnow()

    nyse = get_us_calendar()

    # Convert input date to pandas Timestamp (timezone-naive) for comparison
    if isinstance(date, datetime):
        ts = pd.Timestamp(date)
    else:
        ts = pd.Timestamp(date)

    # Get schedule for a window
    start = ts - pd.Timedelta(days=5)
    end = ts + pd.Timedelta(days=10)
    schedule = nyse.schedule(start_date=start, end_date=end)

    # schedule.index is a DatetimeIndex with timezone info (usually UTC)
    # Convert ts to tz-aware to match
    if schedule.index.tz is not None:
        ts = ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')
    else:
        ts = ts.tz_localize(None) if ts.tz is not None else ts

    # Find first trading day strictly after ts
    future_dates = schedule.index[schedule.index > ts]
    if len(future_dates) > 0:
        next_date = future_dates[0]
        # Return as datetime (timezone-naive or aware as needed)
        return next_date.to_pydatetime()

    # Fallback: if no future date found, add 1 day and try again recursively
    return next_trading_day(date + timedelta(days=1))


def is_trading_day(date: datetime) -> bool:
    """Check if date is a US trading day."""
    nyse = get_us_calendar()
    ts = pd.Timestamp(date)
    start = ts - pd.Timedelta(days=5)
    end = ts + pd.Timedelta(days=5)
    schedule = nyse.schedule(start_date=start, end_date=end)
    # Normalize timezone
    if schedule.index.tz is not None:
        ts = ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')
    else:
        ts = ts.tz_localize(None) if ts.tz is not None else ts
    return ts in schedule.index
