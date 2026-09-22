"""
Management tone scoring.

Two modes:
1. Heuristic / keyword-based (always available, no API key)
2. LLM-based (OpenAI) when OPENAI_API_KEY is set

Scores each transcript on:
- evasiveness (0-10)
- optimism (0-10)
- specificity (0-10)
- confidence (0-10)
- uncertainty (0-10)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

from config import OPENAI_MODEL, TEMPERATURE, TONE_DIMENSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic scorer (no external dependency beyond stdlib)
# ---------------------------------------------------------------------------

EVASIVE_PATTERNS = [
    r"\b(don'?t want to get into|not going to break out|we don'?t disclose)\b",
    r"\b(hard to say|difficult to quantify|not something we break out)\b",
    r"\b(when we are ready|at the appropriate time|stay tuned)\b",
    r"\b(multiple moving pieces|range of scenarios|many variables)\b",
    r"\b(we don'?t provide specific|no specific guidance on)\b",
]

OPTIMISM_PATTERNS = [
    r"\b(extremely|incredibly|very)\s+(optimistic|confident|excited|pleased)\b",
    r"\b(record|all-time high|strong momentum|significant opportunities)\b",
    r"\b(we remain confident|long-term trajectory|massive opportunity)\b",
    r"\b(just getting started|multi-year driver|bright spot)\b",
]

SPECIFICITY_PATTERNS = [
    r"\$[\d,.]+(\s*(billion|million|B|M))?",
    r"\b\d+(\.\d+)?%\b",
    r"\b(Q[1-4]|fiscal\s+\d{4}|next\s+(quarter|year)|by\s+\d{4})\b",
    r"\b(guidance|we expect|we guided)\b",
]

CONFIDENCE_PATTERNS = [
    r"\b(highly confident|very confident|we are confident|strong confidence)\b",
    r"\b(will|expect to|on track to)\b",
]

UNCERTAINTY_PATTERNS = [
    r"\b(maybe|perhaps|potentially|possibly|could be)\b",
    r"\b(uncertain|unclear|hard to predict|macroeconomic)\b",
    r"\b(cautious|cautiously|near-term challenges)\b",
]


def _count_matches(text: str, patterns: List[str]) -> int:
    count = 0
    text_lower = text.lower()
    for p in patterns:
        count += len(re.findall(p, text_lower, flags=re.IGNORECASE))
    return count


def heuristic_score(transcript: str) -> Dict[str, float]:
    """
    Fast, deterministic scoring based on linguistic markers.
    Scaled roughly to 0-10.
    """
    if not transcript:
        return {dim: 5.0 for dim in TONE_DIMENSIONS}

    text = transcript.lower()
    length = max(len(text.split()), 1)

    evasive = _count_matches(text, EVASIVE_PATTERNS)
    optimism = _count_matches(text, OPTIMISM_PATTERNS)
    specificity = _count_matches(text, SPECIFICITY_PATTERNS)
    confidence = _count_matches(text, CONFIDENCE_PATTERNS)
    uncertainty = _count_matches(text, UNCERTAINTY_PATTERNS)

    # Normalize by length (per 500 words) and clamp
    def scale(raw: float, factor: float = 8.0) -> float:
        return max(0.0, min(10.0, (raw / length * 500) * factor))

    scores = {
        "evasiveness": scale(evasive, 12.0),
        "optimism": scale(optimism, 10.0),
        "specificity": scale(specificity, 6.0),
        "confidence": scale(confidence, 9.0),
        "uncertainty": scale(uncertainty, 10.0),
    }
    return scores


# ---------------------------------------------------------------------------
# LLM scorer (OpenAI)
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are a senior equity research analyst specializing in management communication analysis.
Score the following earnings call transcript excerpt on these five dimensions (0-10 scale):

1. evasiveness: How much does management dodge, deflect, or refuse to give concrete answers?
2. optimism: Degree of positive, forward-looking, or promotional language.
3. specificity: Presence of concrete numbers, timelines, KPIs, and clear guidance.
4. confidence: Strength of language and conviction (few hedges).
5. uncertainty: Frequency of hedging, caution, or acknowledgment of unknowns.

Return ONLY a valid JSON object with the five keys and numeric scores. No extra text.

Transcript:
\"\"\"
{transcript}
\"\"\"
"""


def llm_score(transcript: str, max_chars: int = 6000) -> Optional[Dict[str, float]]:
    """Call OpenAI if key is available. Returns None on failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        truncated = transcript[:max_chars]
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": LLM_PROMPT.format(transcript=truncated)},
            ],
            temperature=TEMPERATURE,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.MULTILINE)
        scores = json.loads(content)
        # Validate keys
        return {dim: float(scores.get(dim, 5.0)) for dim in TONE_DIMENSIONS}
    except Exception as e:
        logger.warning(f"LLM scoring failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ToneAnalyzer:
    def __init__(self, prefer_llm: bool = True):
        self.prefer_llm = prefer_llm

    def score(self, transcript: str) -> Dict[str, float]:
        if self.prefer_llm:
            llm_result = llm_score(transcript)
            if llm_result is not None:
                logger.debug("Used LLM scores")
                return llm_result
        logger.debug("Used heuristic scores")
        return heuristic_score(transcript)

    def score_batch(self, transcripts: List[Dict]) -> List[Dict]:
        """Add tone scores to each transcript dict. Returns new list."""
        results = []
        for item in transcripts:
            scored = item.copy()
            scores = self.score(item.get("transcript", ""))
            scored.update(scores)
            # Composite warning signal
            scored["warning_score"] = self._compute_warning(scores)
            results.append(scored)
        return results

    @staticmethod
    def _compute_warning(scores: Dict[str, float]) -> float:
        """
        Higher = more concerning (high evasiveness + high optimism + low specificity).
        Scaled roughly 0-10.
        """
        evasive = scores.get("evasiveness", 5)
        optimism = scores.get("optimism", 5)
        specificity = scores.get("specificity", 5)
        # Low specificity is bad → invert
        warning = (evasive * 0.4) + (optimism * 0.3) + ((10 - specificity) * 0.3)
        return round(min(10.0, max(0.0, warning)), 2)
