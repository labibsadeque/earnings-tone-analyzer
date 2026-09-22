"""
Configuration for the Earnings Call Tone Analyzer.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_TRANSCRIPTS_PATH = DATA_DIR / "sample_transcripts.json"

# Default universe for backtest / demo
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "UNH"
]

# Tone scoring dimensions (0-10 scale)
TONE_DIMENSIONS = [
    "evasiveness",      # higher = more evasive / dodging questions
    "optimism",         # higher = more positive / forward-looking hype
    "specificity",      # higher = more concrete numbers, timelines, KPIs
    "confidence",       # higher = stronger language, fewer hedges
    "uncertainty"       # higher = more hedging, "maybe", "potentially"
]

# Signal construction
# Example: High evasiveness + High optimism + Low specificity = "Warning" signal
WARNING_THRESHOLDS = {
    "evasiveness": 6.5,
    "optimism": 7.0,
    "specificity": 4.0,   # low specificity is bad
}

# Backtest settings
FORWARD_DAYS = 60
RISK_FREE_RATE = 0.04  # annualized, for simple excess return calc

# LLM settings (optional)
OPENAI_MODEL = "gpt-4o-mini"  # cheap & good for this task
MAX_TOKENS_PER_CALL = 1200
TEMPERATURE = 0.1

# Cache
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
