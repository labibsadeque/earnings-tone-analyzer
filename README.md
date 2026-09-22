# AI-Powered Earnings Call "Early Warning Signal" Analyzer

**Quantitative research prototype for fundamental / event-driven strategies.**

This project implements an end-to-end NLP + market-data pipeline that:

1. Loads earnings-call transcripts (sample data included; real APIs pluggable)
2. Scores management tone on five dimensions: **evasiveness, optimism, specificity, confidence, uncertainty**
3. Pulls subsequent **60-day stock returns** via `yfinance`
4. Backtests whether tone-shift signals (especially high optimism + high evasiveness + low specificity) predict weaker forward returns
5. Surfaces everything in an interactive **Streamlit dashboard**

---

## Investment Thesis

Classic sentiment analysis on earnings calls often fails because management is trained to sound positive. The more useful signal is the **gap** between:

- What management *says* (tone, hedging, specificity), and  
- What *actually happens* to the stock over the following weeks.

Empirical patterns observed by many fundamental desks:

- **High optimism + high evasiveness + low specificity** frequently precedes underperformance.  
  Management is “talking their book” while avoiding concrete commitments.
- Purely positive language with high specificity and low hedging is less informative (or even mildly bullish).
- Tone *shifts* quarter-over-quarter (rising evasiveness while optimism stays elevated) are especially interesting for event-driven books.

This tool quantifies those linguistic features and measures their historical relationship to forward returns so a portfolio manager can decide whether the signal is worth further research or live monitoring.

> **Disclaimer**: This is a research prototype, not investment advice. Sample size in the demo is small. Always conduct your own due diligence and risk management.

---

## Quick Start (Offline Demo)

```bash
# 1. Clone / copy the project
cd earnings-tone-analyzer

# 2. Create environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) set OpenAI key for LLM scoring
export OPENAI_API_KEY=sk-...

# 4. Launch dashboard
streamlit run app.py
```

The dashboard runs entirely offline using the included sample transcripts.  
Toggle “Prefer OpenAI LLM scoring” only if you have a key.

---

## Architecture

```
transcript_loader.py  →  tone_analyzer.py  →  price_fetcher.py  →  backtester.py
         ↓                      ↓                    ↓                  ↓
   sample JSON / API      heuristic or LLM      yfinance 60d ret    stats + OLS
                                                         ↓
                                                   Streamlit UI
```

### Key Modules

| Module | Responsibility |
|--------|----------------|
| `transcript_loader.py` | Load sample JSON or (when keys present) call external transcript APIs |
| `tone_analyzer.py` | Heuristic keyword + regex scorer + optional OpenAI GPT scoring |
| `price_fetcher.py` | Download adjusted close prices and compute forward total returns |
| `backtester.py` | Event study, signal generation, correlations, simple OLS |
| `app.py` | Interactive dashboard |

### Tone Dimensions (0–10)

- **Evasiveness** – dodging questions, “we don’t break that out”, “when we’re ready”
- **Optimism** – “extremely confident”, “massive opportunity”, record language
- **Specificity** – concrete $ figures, %, dates, explicit guidance
- **Confidence** – strong verbs, few hedges
- **Uncertainty** – “maybe”, “macro”, “hard to predict”

A composite **warning_score** is a weighted combination that rises when optimism and evasiveness are high while specificity is low.

---

## Extending to Production

### 1. Real Transcript Sources

Popular options (most require paid plans for full historical coverage):

- [earningscalls.dev / RapidAPI](https://earningscalls.dev/)
- [Financial Modeling Prep](https://site.financialmodelingprep.com/)
- [Finnhub](https://finnhub.io/)
- [EarningsCall.biz](https://earningscall.biz/)
- Hugging Face datasets (e.g. S&P 500 transcripts) for offline research

Implement the fetch logic inside `TranscriptLoader._fetch_from_apis`.

### 2. Better NLP

- Replace the heuristic scorer with a fine-tuned classifier (RoBERTa / FinBERT) on labeled evasiveness data.
- Use speaker diarization to score only management answers (ignore analyst questions).
- Track quarter-over-quarter *changes* in scores rather than absolute levels.

### 3. Stronger Backtest

- Expand universe (S&P 500 / Russell 3000)
- Control for earnings surprise, sector, market beta, size
- Use proper event-study methodology (abnormal returns vs. market or industry)
- Walk-forward or purged cross-validation to avoid look-ahead bias
- Position sizing / portfolio simulation with transaction costs

### 4. Live Monitoring

- Schedule daily / post-earnings jobs
- Alert when a new transcript scores above a warning threshold
- Feed into an existing research dashboard or risk system

---

## Configuration

Edit `config.py`:

- `DEFAULT_TICKERS`
- `WARNING_THRESHOLDS`
- `FORWARD_DAYS`
- OpenAI model name

Environment variables:

```bash
export OPENAI_API_KEY=...
export FINNHUB_API_KEY=...
export RAPIDAPI_KEY=...
export FMP_API_KEY=...
```

---

## Project Layout

```
earnings-tone-analyzer/
├── README.md
├── requirements.txt
├── config.py
├── app.py                 # Streamlit entry point
├── data/
│   └── sample_transcripts.json
└── src/
    ├── transcript_loader.py
    ├── tone_analyzer.py
    ├── price_fetcher.py
    ├── backtester.py
    └── utils.py
```

---

## Example Research Questions This Tool Helps Answer

- Do high-warning-score calls underperform low-warning-score calls on average?
- Which single dimension (evasiveness vs. specificity) has the strongest univariate relationship with 60-day returns?
- Is the signal stronger in certain sectors (tech, consumer discretionary)?
- Does the signal survive after controlling for the earnings surprise itself?

---

## License & Attribution

This is a demonstration project created for quant / hedge-fund application purposes.  
Sample transcripts are synthetic / illustrative.  
Use at your own risk; past linguistic patterns do not guarantee future performance.
