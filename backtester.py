"""
Simple event-study / signal backtest.

Compares tone-derived warning signals against subsequent 60-day stock returns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from config import FORWARD_DAYS, WARNING_THRESHOLDS

logger = logging.getLogger(__name__)


class ToneBacktester:
    def __init__(self, forward_days: int = FORWARD_DAYS):
        self.forward_days = forward_days
        self.return_col = f"forward_return_{forward_days}d"

    def to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if self.return_col not in df.columns:
            raise ValueError(f"Missing {self.return_col}. Did you run price attachment?")
        # Drop rows without returns
        df = df.dropna(subset=[self.return_col]).copy()
        df["call_date"] = pd.to_datetime(df["call_date"])
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary warning signal based on thresholds.
        Also keep continuous warning_score.
        """
        df = df.copy()
        df["is_warning"] = (
            (df["evasiveness"] >= WARNING_THRESHOLDS["evasiveness"])
            & (df["optimism"] >= WARNING_THRESHOLDS["optimism"])
            & (df["specificity"] <= WARNING_THRESHOLDS["specificity"])
        )
        return df

    def summary_stats(self, df: pd.DataFrame) -> Dict:
        """Key performance statistics of the signal."""
        if df.empty:
            return {}

        overall_mean = df[self.return_col].mean()
        overall_std = df[self.return_col].std()

        warn = df[df["is_warning"]]
        non_warn = df[~df["is_warning"]]

        stats = {
            "n_events": len(df),
            "n_warning": len(warn),
            "n_non_warning": len(non_warn),
            "mean_return_all": overall_mean,
            "mean_return_warning": warn[self.return_col].mean() if len(warn) else np.nan,
            "mean_return_non_warning": non_warn[self.return_col].mean() if len(non_warn) else np.nan,
            "std_return_all": overall_std,
            "warning_hit_rate": (warn[self.return_col] < 0).mean() if len(warn) else np.nan,
            "avg_warning_score": df["warning_score"].mean(),
        }

        # Simple t-ish difference
        if len(warn) > 1 and len(non_warn) > 1:
            diff = stats["mean_return_warning"] - stats["mean_return_non_warning"]
            stats["return_diff_warning_vs_rest"] = diff
        else:
            stats["return_diff_warning_vs_rest"] = np.nan

        return stats

    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pearson correlation of each tone dimension with forward returns."""
        cols = [
            "evasiveness",
            "optimism",
            "specificity",
            "confidence",
            "uncertainty",
            "warning_score",
            self.return_col,
        ]
        available = [c for c in cols if c in df.columns]
        corr = df[available].corr()[self.return_col].drop(self.return_col)
        return corr.sort_values().to_frame("correlation_with_forward_return")

    def regression(self, df: pd.DataFrame) -> Dict:
        """OLS: forward return ~ tone dimensions."""
        features = ["evasiveness", "optimism", "specificity", "confidence", "uncertainty"]
        X = df[features].fillna(df[features].mean())
        y = df[self.return_col]
        if len(df) < 5:
            return {"r2": np.nan, "coefficients": {}}
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        return {
            "r2": r2_score(y, preds),
            "coefficients": dict(zip(features, model.coef_)),
            "intercept": model.intercept_,
        }

    def run(self, records: List[Dict]) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, Dict]:
        """
        Full pipeline:
        - DataFrame
        - Summary stats
        - Correlation table
        - Regression results
        """
        df = self.to_dataframe(records)
        df = self.generate_signals(df)
        stats = self.summary_stats(df)
        corr = self.correlation_analysis(df)
        reg = self.regression(df)
        logger.info(f"Backtest complete: {stats.get('n_events', 0)} events")
        return df, stats, corr, reg
