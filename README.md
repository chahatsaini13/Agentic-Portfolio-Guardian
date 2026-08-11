# Agentic AI Portfolio Guardian

Retail investors usually buy a stock based on a thesis (e.g. "this company
will benefit from EV adoption"), but rarely revisit whether that thesis
still holds as news, fundamentals, and market conditions change. This
project builds a multi-agent system that continuously re-checks the
original reasoning behind each holding using live market data, financial
news, and portfolio composition — surfacing what has changed and why,
**without issuing direct buy/sell recommendations**.

## Architecture

```mermaid
graph TD
    A[Investment Thesis Agent] --> O[Orchestrator Agent]
    B[Portfolio Health Agent] --> O
    C[Market Intelligence Agent] --> O
    D[Early Warning Agent] --> O
    O --> U[User-facing Insight]
```

Four specialized agents coordinated by a central Orchestrator, built with
LangGraph so state and control flow between agents stays explicit and
traceable.

| Agent | Role | Status |
|---|---|---|
| Investment Thesis Agent | Tracks whether the original reason a user invested still holds against current data | Completed |
| Portfolio Health Agent | Evaluates diversification and risk exposure across the full portfolio | In progress |
| Market Intelligence Agent | Reads live news/market updates and summarizes what's relevant to each holding | Not started |
| Early Warning Agent | Flags red flags: management changes, litigation, credit downgrades | Not started |
| Orchestrator Agent | Combines all four agents' outputs into one coherent, user-facing insight | Not started |

## Tech stack
- Python 3.x
- LangGraph (agent orchestration)
- LLM: Ollama v0.32.4 (Local) — `llama3.2:latest`
- Embeddings: sentence-transformers, contrastively fine-tuned relevance model
  hosted on Hugging Face Hub
- Data: yfinance, NewsAPI
- Dashboard/UI: TBD — agents output structured JSON; UI framework to be decided

## Relevance filtering
News is filtered for relevance to each investment thesis using embedding
similarity rather than keyword matching, which fails on "hard negative"
cases (news about the same company that's unrelated to the specific
thesis). A contrastively fine-tuned model
(`navneet11/contrastive-relevance-v1`, hosted privately on Hugging Face
Hub) replaced an initial generic-embedding placeholder after evaluation
showed clean separation between relevant and irrelevant news, including
on hard negatives. See `docs/decisions/0002-...MD` and
`docs/decisions/0003-...MD` for the full methodology and evaluation.

## Data sources
- Live market/news: yfinance (NSE/BSE), NewsAPI
- Financial PhraseBank (~4,840 expert-labeled sentences) — sentiment benchmark
- Historical NSE data (1,700+ stocks, 1990–2021) — backtesting

## Evaluation
- Sentiment classification accuracy benchmarked against Financial PhraseBank
- Thesis-flip detection validated against historical NSE data (1990–2021)

## Limitations
- NewsAPI free tier restricts article recency and request volume
- yfinance can be inconsistent/rate-limited for NSE/BSE tickers
- When both fundamentals and news data are sparse or unavailable, the LLM
  may produce confident-sounding but unsubstantiated claims rather than
  flagging insufficient data (observed during testing, not yet mitigated)
- Contrastive relevance model trained on a small dataset (8 investment
  theses) - strong held-out evaluation results, but not yet validated at scale
- Not intended as investment advice; outputs are informational only

## Decision log
Every meaningful design decision is recorded in `docs/decisions/` as an
ADR (Architecture Decision Record) — see `docs/decisions/template.md` for
the format. Read these before making changes; they capture *why*, not just
*what*.

## Project structure
```
src/agents/            individual agent implementations
docs/decisions/         ADRs — one file per decision
requirements.txt
.env.example            copy to .env and fill in your API keys
```

## Setup
```
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
python src/agents/investment_thesis_agent.py
```
