"""
Stock price & forward-return utilities using yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.utils import forward_return, parse_date

logger = logging.getLogger(__name__)


class PriceFetcher:
    def __init__(self, cache: Optional[Dict] = None):
        self._cache: Dict[str, pd.DataFrame] = cache or {}

    def get_history(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        if ticker in self._cache:
            df = self._cache[ticker]
        else:
            try:
                t = yf.Ticker(ticker)
                # Pull a generous window so we can compute 60-day returns
                df = t.history(period="5y", auto_adjust=True)
                if df.empty:
                    logger.warning(f"No price data for {ticker}")
                    return pd.DataFrame()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                self._cache[ticker] = df
            except Exception as e:
                logger.error(f"yfinance error for {ticker}: {e}")
                return pd.DataFrame()

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def get_forward_return(
        self,
        ticker: str,
        call_date: datetime,
        days: int = 60,
    ) -> Optional[float]:
        if call_date is None:
            return None
        # Start looking a few days before the call in case of after-hours
        start = call_date - timedelta(days=5)
        end = call_date + timedelta(days=days + 30)
        hist = self.get_history(ticker, start=start, end=end)
        if hist.empty or "Close" not in hist.columns:
            return None
        return forward_return(hist["Close"], call_date, days=days)

    def attach_returns(
        self,
        scored_transcripts: List[Dict],
        days: int = 60,
    ) -> List[Dict]:
        """Add forward_return_{days}d field to each record."""
        results = []
        for item in scored_transcripts:
            row = item.copy()
            call_date = item.get("call_date")
            if isinstance(call_date, str):
                call_date = parse_date(call_date)
            ret = self.get_forward_return(item["ticker"], call_date, days=days)
            row[f"forward_return_{days}d"] = ret
            results.append(row)
        return results
