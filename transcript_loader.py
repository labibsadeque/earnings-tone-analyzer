"""
Transcript loading layer.

Supports:
1. Local sample / cached JSON (always works offline)
2. Optional real APIs (Finnhub, RapidAPI, FMP, etc.) when keys are present
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

from config import SAMPLE_TRANSCRIPTS_PATH
from src.utils import clean_transcript_text, load_json, parse_date

logger = logging.getLogger(__name__)


class TranscriptLoader:
    """Unified interface for earnings call transcripts."""

    def __init__(self, use_sample: bool = True):
        self.use_sample = use_sample
        self._sample_cache: Optional[List[Dict]] = None

    def _load_sample(self) -> List[Dict]:
        if self._sample_cache is None:
            raw = load_json(SAMPLE_TRANSCRIPTS_PATH)
            cleaned = []
            for item in raw:
                item = item.copy()
                item["transcript"] = clean_transcript_text(item.get("transcript", ""))
                item["call_date"] = parse_date(item.get("call_date", ""))
                cleaned.append(item)
            self._sample_cache = cleaned
            logger.info(f"Loaded {len(cleaned)} sample transcripts")
        return self._sample_cache

    def get_transcripts(
        self,
        tickers: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> List[Dict]:
        """
        Return list of transcript dicts with keys:
        ticker, company, fiscal_year, fiscal_quarter, call_date, transcript
        """
        if self.use_sample or not self._has_api_keys():
            data = self._load_sample()
        else:
            data = self._fetch_from_apis(tickers or [])

        # Filter
        if tickers:
            tickers_upper = {t.upper() for t in tickers}
            data = [d for d in data if d["ticker"].upper() in tickers_upper]
        if start_year:
            data = [d for d in data if d.get("fiscal_year", 0) >= start_year]
        if end_year:
            data = [d for d in data if d.get("fiscal_year", 0) <= end_year]

        return data

    def _has_api_keys(self) -> bool:
        return bool(
            os.getenv("FINNHUB_API_KEY")
            or os.getenv("RAPIDAPI_KEY")
            or os.getenv("FMP_API_KEY")
        )

    def _fetch_from_apis(self, tickers: List[str]) -> List[Dict]:
        """Placeholder for real API integrations. Falls back to sample on failure."""
        results = []
        for ticker in tickers:
            try:
                # Example: Finnhub (requires paid plan for full transcripts)
                key = os.getenv("FINNHUB_API_KEY")
                if key:
                    # Finnhub transcript endpoint is limited; this is illustrative
                    url = f"https://finnhub.io/api/v1/stock/transcripts"
                    # Real implementation would paginate / filter by year-quarter
                    logger.warning(
                        "Real Finnhub transcript fetch not fully implemented in demo. "
                        "Using sample data."
                    )
            except Exception as e:
                logger.error(f"API fetch failed for {ticker}: {e}")
        # Always fall back so the pipeline never breaks
        return self._load_sample()

    def get_available_tickers(self) -> List[str]:
        data = self.get_transcripts()
        return sorted({d["ticker"] for d in data})


# Convenience function
def load_transcripts(
    tickers: Optional[List[str]] = None,
    use_sample: bool = True,
) -> List[Dict]:
    loader = TranscriptLoader(use_sample=use_sample)
    return loader.get_transcripts(tickers=tickers)
