"""
Utility helpers: logging, cleaning, date handling, caching.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def clean_transcript_text(text: str) -> str:
    """Basic cleaning: normalize whitespace, remove artifacts."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[.*?\]", "", text)  # remove stage directions
    text = text.strip()
    return text


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse common date formats safely."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(date_str[:10], fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except Exception:
        logger.warning(f"Could not parse date: {date_str}")
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def forward_return(
    prices: pd.Series,
    start_date: datetime,
    days: int = 60,
) -> Optional[float]:
    """
    Compute simple total return from the first trading day on/after start_date
    to approximately `days` calendar days later.
    """
    if prices.empty:
        return None
    prices = prices.sort_index()
    # Find first available price on or after start_date
    mask = prices.index >= pd.Timestamp(start_date)
    if not mask.any():
        return None
    start_price = prices.loc[mask].iloc[0]
    start_idx = prices.loc[mask].index[0]

    end_target = start_idx + timedelta(days=days)
    end_mask = prices.index >= end_target
    if not end_mask.any():
        # use last available
        end_price = prices.iloc[-1]
    else:
        end_price = prices.loc[end_mask].iloc[0]

    if start_price <= 0:
        return None
    return (end_price / start_price) - 1.0
