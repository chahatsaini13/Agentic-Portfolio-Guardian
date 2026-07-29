# Project Roadmap — Agentic AI Portfolio Guardian

**Duration:** 9–10 weeks
**Team:** Member 1 (Chahat Saini), Member 2 (Navneet Sah)

Status legend: ✅ Completed · 🔄 In Progress · ⬜ Not Started

---

## Week 1 — Setup & Foundations
| Task | Owner |
|---|---|
| Repo structure, README, .gitignore, .env.example, requirements.txt | Member 1 |
| Set up Ollama locally and verify prompt/response pipeline | Member 1 |
| Research yfinance and NewsAPI, explore available endpoints | Member 2 |
| Learn LangGraph basics (nodes, edges, state management) | Both |
**Status: ✅ Completed**

## Week 2 — Investment Thesis Agent (Initial Prototype) 
| Task | Owner |
|---|---|
| Build stock price & fundamentals fetch using yfinance | Member 1 |
| Build basic news retrieval using NewsAPI | Member 1 |
| Design initial Ollama prompt to compare investment thesis with current market data | Member 1 |
| Run the prototype on 2–3 sample stocks | Member 1 |
| Prepare a working demo for supervisor review | Both |
**Pipeline:**
```
Get stock price & fundamentals from yfinance → Get relevant news using NewsAPI
→ Build combined prompt (thesis + fundamentals + news) → Send prompt to local Ollama model
→ Parse model response into structured JSON
```
**Status: ✅ Completed**

## Week 3 — Investment Thesis Agent (Refinement)
| Task | Owner |
|---|---|
| Improve news filtering and preprocessing | Member 2 |
| Structure agent output (Thesis Holds / Weakened / Broken + reasoning) | Member 1 |
| Expand testing to 5–10 sample stocks and log results | Both |
| Write first ADR: Why Ollama + Prompt Design | Member 2 |
**Pipeline:**
```
Get stock price & fundamentals from yfinance → Get news using NewsAPI
→ Filter out irrelevant/noisy news articles → Build combined prompt
→ Send prompt to local Ollama model → Parse model response into JSON
→ Validate output matches expected structure
```
**Status: ⬜ Not Started**

## Week 4 — Portfolio Health Agent
| Task | Owner |
|---|---|
| Load sample portfolio (holdings, sectors, weights) | Member 2 |
| Compute diversification/concentration metrics | Member 2 |
| Flag over-exposure to a sector/stock | Member 1 |
| Output structured risk summary | Member 1 |
**Pipeline:**
```
Load sample portfolio (holdings, sectors, weights) → Compute weight of each sector/stock
→ Compute diversification/concentration score → Flag stocks/sectors that are over-exposed
→ Build structured risk summary
```
**Status: ⬜ Not Started**

## Week 5 — Market Intelligence Agent
| Task | Owner |
|---|---|
| Pull live market/news updates per holding | Member 1 |
| Summarize relevant developments using LLM | Member 2 |
| Benchmark sentiment tagging against Financial PhraseBank | Member 2 |
**Pipeline:**
```
Get recent news for each holding → Summarize key developments using local LLM
→ Tag sentiment of each news item → Check sentiment accuracy against Financial PhraseBank
→ Build structured summary output
```
**Status: ⬜ Not Started**

## Week 6 — Early Warning Agent 
| Task | Owner |
|---|---|
| Detect red-flag events (management change, litigation, downgrades) | Member 1 |
| Score severity/urgency of each flag | Member 2 |
| Output structured alerts | Both |
| Prepare demo of all 4 agents (individually) for supervisor review | Both |
**Pipeline:**
```
Get recent news for each holding → Detect red-flag events (management change, litigation, downgrades)
→ Score severity/urgency of each flag using LLM → Build structured alert output
```
**Status: ⬜ Not Started**

## Week 7 — Orchestrator Agent (LangGraph Integration)
| Task | Owner |
|---|---|
| Define LangGraph state graph connecting all 4 agents | Member 1 |
| Merge outputs into one coherent per-stock + per-portfolio insight | Member 2 |
| Handle agent failures/timeouts gracefully | Both |
**Pipeline:**
```
Define shared state structure for LangGraph → Add each of the 4 agents as a graph node
→ Connect nodes in the right order/logic → Run the full graph
→ Merge all agent outputs into one combined insight → Handle failures/timeouts gracefully
```
**Status: ⬜ Not Started**

## Week 8 — Dashboard / UI Layer
| Task | Owner |
|---|---|
| Decide + set up UI framework (Streamlit) | Member 2 |
| Design views: thesis status, portfolio health, alerts feed | Member 1 |
| Wire Orchestrator's JSON output into the UI | Both |
**Pipeline:**
```
Load Orchestrator's combined JSON output → Display thesis status per stock
→ Display portfolio health/risk view → Display red-flag alerts feed
→ Assemble everything into one dashboard
```
**Status: ⬜ Not Started**

## Week 9 — Evaluation 
| Task | Owner |
|---|---|
| Validate sentiment agent accuracy against Financial PhraseBank | Member 2 |
| Backtest thesis-flip detection on historical NSE data (1990–2021) | Member 1 |
| Document results in `docs/decisions/` | Both |
| Full end-to-end demo for supervisor review | Both |
**Pipeline:**
```
Load evaluation dataset (Financial PhraseBank / historical NSE data)
→ Run agent on the dataset → Compute accuracy/backtest metrics
→ Log results and findings as an ADR
```
**Status: ⬜ Not Started**

## Week 10 — Polish & Submission (buffer week)
| Task | Owner |
|---|---|
| Clean up code, add docstrings/comments | Both |
| Finalize README with screenshots/demo GIF | Member 1 |
| Record demo video (if required for viva) | Member 2 |
| Final push + tag release (v1.0) | Both |
**Status: ⬜ Not Started**

---

