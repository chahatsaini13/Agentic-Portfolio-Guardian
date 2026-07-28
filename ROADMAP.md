# Project Roadmap — Agentic AI Portfolio Guardian

**Duration:** 9–10 weeks
**Team:** Member 1 (Chahat Saini), Member 2 (Navneet Sah)

Status legend: ✅ Completed · 🔄 In Progress · ⬜ Not Started

---

## Week 1 — Setup & Foundations
| Task | Owner |
|---|---|
| Repo structure, README, .gitignore, .env.example, requirements.txt | Member 1 |
| Set up Ollama locally, test basic prompt/response pipeline | Member 1 |
| Research yfinance + NewsAPI, pull sample data for 2–3 stocks | Member 2 |
| Read up on LangGraph basics (nodes, edges, state) — shared learning | Both |
**Status: ✅ Completed**

## Week 2 — Investment Thesis Agent (Core Logic)
| Task | Owner |
|---|---|
| Build price/fundamentals fetch via yfinance | Member 1 |
| Build news fetch + cleaning via NewsAPI | Member 2 |
| Design prompt for Ollama: compare thesis vs current data | Member 1 |
| Test agent end-to-end on 3–5 sample stocks, log outputs | Both |
**Status: ✅ Completed**

## Week 3 — Investment Thesis Agent (Refine) + Checkpoint 1
| Task | Owner |
|---|---|
| Structure agent output (thesis holds/weakened/broken + reasoning) | Member 1 |
| Write first ADR: why Ollama + this prompt design | Member 2 |
| Prepare demo of Thesis Agent for supervisor review | Both |
**Status: ⬜ Not Started**

## Week 4 — Portfolio Health Agent
| Task | Owner |
|---|---|
| Load sample portfolio (holdings, sectors, weights) | Member 2 |
| Compute diversification/concentration metrics | Member 2 |
| Flag over-exposure to a sector/stock | Member 1 |
| Output structured risk summary | Member 1 |
**Status: ⬜ Not Started**

## Week 5 — Market Intelligence Agent
| Task | Owner |
|---|---|
| Pull live market/news updates per holding | Member 1 |
| Summarize relevant developments using LLM | Member 2 |
| Benchmark sentiment tagging against Financial PhraseBank | Member 2 |
**Status: ⬜ Not Started**

## Week 6 — Early Warning Agent + Checkpoint 2
| Task | Owner |
|---|---|
| Detect red-flag events (management change, litigation, downgrades) | Member 1 |
| Score severity/urgency of each flag | Member 2 |
| Output structured alerts | Both |
| Prepare demo of all 4 agents (individually) for supervisor review | Both |
**Status: ⬜ Not Started**

## Week 7 — Orchestrator Agent (LangGraph Integration)
| Task | Owner |
|---|---|
| Define LangGraph state graph connecting all 4 agents | Member 1 |
| Merge outputs into one coherent per-stock + per-portfolio insight | Member 2 |
| Handle agent failures/timeouts gracefully | Both |
**Status: ⬜ Not Started**

## Week 8 — Dashboard / UI Layer
| Task | Owner |
|---|---|
| Decide + set up UI framework (Streamlit) | Member 2 |
| Design views: thesis status, portfolio health, alerts feed | Member 1 |
| Wire Orchestrator's JSON output into the UI | Both |
**Status: ⬜ Not Started**

## Week 9 — Evaluation + Checkpoint 3
| Task | Owner |
|---|---|
| Validate sentiment agent accuracy against Financial PhraseBank | Member 2 |
| Backtest thesis-flip detection on historical NSE data (1990–2021) | Member 1 |
| Document results in `docs/decisions/` | Both |
| Full end-to-end demo for supervisor review | Both |
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

