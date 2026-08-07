# Project Roadmap — Agentic AI Portfolio Guardian

**Duration:** 9–10 weeks
**Team:** Member 1 (Chahat Saini), Member 2 (Navneet Sah)

Status legend: ✅ Completed · 🔄 In Progress · ⬜ Not Started

## Contrastive Learning Integration
Per supervisor's suggestion, we're incorporating contrastive learning at
points where the underlying task is fundamentally about similarity or
separation, rather than bolting it on everywhere it doesn't fit:

- **News relevance filtering (Week 3):** learn embeddings that pull
  thesis-relevant news closer together and push irrelevant news further
  apart, instead of relying on keyword overlap alone.
- **Sentiment tagging (Week 5):** fine-tune sentence embeddings
  contrastively (SimCSE-style, using Financial PhraseBank as labeled
  anchor pairs) so similar-sentiment financial text clusters together in
  embedding space — expected to outperform generic off-the-shelf
  embeddings on our benchmark.
- **Red-flag detection (Week 6):** treat this as anomaly detection —
  contrastively train embeddings so "routine business news" and
  "red-flag news" (management change, litigation, downgrades) separate
  into distinguishable clusters, supplementing rule-based keyword
  matching with a learned signal.
- **Evaluation (Week 9):** measure whether the contrastive embeddings
  actually improved separation/accuracy over the baseline (keyword-only /
  generic embeddings), using Financial PhraseBank labels as ground truth.
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

## Week 3 — Investment Thesis Agent (Refinement + Contrastive Filtering)
| Task | Owner |
|---|---|
| Collect thesis-relevant vs irrelevant news pairs for training data | Member 2 |
| Train/fine-tune contrastive embedding model on these pairs | Member 2 |
| Replace keyword-based news filtering with embedding-similarity filtering | Member 1 |
| Structure agent output (Thesis Holds / Weakened / Broken + reasoning) | Member 1 |
| Expand testing to 5–10 sample stocks and log results | Both |
| Write ADR: Why Ollama + Prompt Design, and why contrastive filtering over keyword matching | Member 2 |
**Pipeline:**
```
Get stock price & fundamentals from yfinance → Get news using NewsAPI
→ Embed thesis + news using contrastively-trained encoder
→ Keep news above similarity threshold, discard the rest → Build combined prompt
→ Send prompt to local Ollama model → Parse model response into JSON
→ Validate output matches expected structure
```
**Status: ⬜ Not Started**

## Week 4 — Portfolio Health Agent
| Task | Owner |
|---|---|
| Load sample portfolio (holdings, sectors, weights) | Member 1 |
| Compute diversification/concentration metrics | Member 1 |
| Flag over-exposure to a sector/stock | Member 2 |
| Output structured risk summary | Member 2 |
**Pipeline:**
```
Load sample portfolio (holdings, sectors, weights) → Compute weight of each sector/stock
→ Compute diversification/concentration score → Flag stocks/sectors that are over-exposed
→ Build structured risk summary
```
**Status: ⬜ Not Started**

## Week 5 — Market Intelligence Agent (with Contrastive Sentiment Tagging)
| Task | Owner |
|---|---|
| Pull live market/news updates per holding | Member 1 |
| Fine-tune sentence embeddings contrastively using Financial PhraseBank | Member 2 |
| Tag sentiment using contrastive embedding space (nearest-cluster match) | Member 2 |
| Summarize relevant developments using LLM | Member 1 |
| Compare contrastive tagging accuracy vs generic embedding baseline | Both |
**Pipeline:**
```
Get recent news for each holding → Embed news using contrastively fine-tuned
sentence encoder → Tag sentiment via nearest cluster in embedding space
→ Compare against Financial PhraseBank labels for accuracy
→ Summarize key developments using local LLM → Build structured summary output
```
**Status: ⬜ Not Started**

## Week 6 — Early Warning Agent (Contrastive Anomaly Detection)
| Task | Owner |
|---|---|
| Label sample of "routine" vs "red-flag" news for training pairs | Member 1 |
| Train contrastive embeddings to separate routine vs red-flag news | Member 2 |
| Combine contrastive signal with rule-based keyword detection | Member 1 |
| Score severity/urgency of each flag using LLM | Member 2 |
| Output structured alerts | Both |
| Prepare demo of all 4 agents (individually) for supervisor review | Both |
**Pipeline:**
```
Get recent news for each holding → Embed news using contrastively trained encoder
→ Flag news that falls in the "red-flag" cluster (anomaly signal)
→ Cross-check against rule-based keyword detection → Score severity/urgency using LLM
→ Build structured alert output
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
*Future enhancement (post-MVP): Add a self-reflection step to agent outputs (Reflexion-style): after Ollama returns a verdict, a second pass checks whether the reasoning actually cites specific data/news points vs being generic, and re-prompts if weak. Natural fit once Orchestrator exists.*
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

## Week 9 — Evaluation (incl. Contrastive Embedding Quality)
| Task | Owner |
|---|---|
| Validate sentiment agent accuracy against Financial PhraseBank | Member 2 |
| Measure contrastive embedding separation quality (e.g. cluster purity/silhouette score) vs generic embedding baseline | Member 2 |
| Backtest thesis-flip detection on historical NSE data (1990–2021) | Member 1 |
| Document results in `docs/decisions/` | Both |
| Full end-to-end demo for supervisor review | Both |
**Pipeline:**
```
Load evaluation dataset (Financial PhraseBank / historical NSE data)
→ Run agent on the dataset → Compute accuracy/backtest metrics
→ Compare contrastive-embedding results vs generic-embedding baseline
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
