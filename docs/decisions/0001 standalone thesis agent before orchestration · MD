# ADR 0001: Build Investment Thesis Agent as a standalone script before LangGraph integration

## Status
Accepted

## Context
The Portfolio Guardian project will eventually orchestrate multiple agents
via LangGraph. Building the orchestration layer and the first agent's logic
at the same time makes it hard to tell which layer a bug belongs to.

## Decision
Implement the Investment Thesis Agent as a single, dependency-light Python
script (`src/agents/investment_thesis_agent.py`) that:
- takes ticker + thesis as CLI args
- fetches stock fundamentals using yfinance
- fetches recent company and thesis-related news using NewsAPI
- calls a local Ollama model directly via its HTTP API
- returns a plain dict/JSON result

No LangGraph nodes, state graphs, or tool-calling abstractions are used yet.

## Consequences
- Faster to build and debug in isolation; failures are clearly isolated to
  either data retrieval, prompt construction, model output, or JSON parsing.
- The `run(ticker, thesis) -> dict` function is already shaped like a
  LangGraph node function, so wrapping it later should be close to
  mechanical rather than requiring a rewrite.
- Secrets (`NEWSAPI_KEY`, `OLLAMA_MODEL`) are loaded via `.env` using
  `python-dotenv`, making later migration to a multi-agent architecture
  straightforward.
- The standalone implementation was validated on multiple sample stocks
  (e.g., Tata Motors, Reliance Industries, and Infosys) before integration.

## Validation

The standalone prototype was tested on multiple stocks to verify the
end-to-end workflow (fundamentals → news retrieval → LLM reasoning → JSON
output).

### Test Case 1

**Ticker:** TMCV.NS (Tata Motors)

**Investment Thesis:**
"I bought this because EV adoption will drive demand."

**Result:**
```json
{
  "ticker": "TMCV.NS",
  "thesis_status": "BROKEN",
  "reasoning": "The thesis that EV adoption will drive demand for TATA MOTORS LIMITED's stock is weakened by the recent news on resale fears and price parity with ICE vehicles. Vivek Srivatsa, CCO of Tata Passenger Electric Mobility, mentions treating EVs like gadgets rather than cars, suggesting affordability and resale value remain important concerns. These developments weaken the original investment thesis.",
  "key_changes": [
    "Resale concerns around EVs",
    "Price parity with ICE vehicles remains a challenge"
  ]
}
```

---

### Test Case 2

**Ticker:** RELIANCE.NS (Reliance Industries)

**Investment Thesis:**
"I bought this because Jio expansion will drive stable growth."

**Result:**
```json
{
  "ticker": "RELIANCE.NS",
  "thesis_status": "HOLDS",
  "reasoning": "The thesis still holds as Jio's expansion continues to support stable growth while refining margins have strengthened, improving the company's outlook.",
  "key_changes": [
    "Refining margins improved",
    "Continued Jio expansion"
  ]
}
```

---

### Test Case 3

**Ticker:** INFY.NS (Infosys)

**Investment Thesis:**
"I bought this because AI services will accelerate revenue growth."

**Result:**
```json
{
  "ticker": "INFY.NS",
  "thesis_status": "WEAKENING",
  "reasoning": "The thesis is weakening because AI revenue growth has been lower than expected and the company recently faced a regulatory penalty related to its time-tracking system. While AI remains a growth area, current evidence suggests a slower acceleration than anticipated.",
  "key_changes": [
    "AI revenue growth at 8.2%",
    "Time-tracking system penalty"
  ]
}
```

**Summary**

These test cases demonstrate that the prototype successfully:
- Retrieves stock fundamentals using yfinance.
- Retrieves recent company and thesis-related news using NewsAPI.
- Generates an investment thesis evaluation using a local Ollama LLM.
- Returns structured JSON output for downstream integration into LangGraph.