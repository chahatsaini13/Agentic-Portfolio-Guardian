"""
Agentic Portfolio Guardian - Streamlit MVP 

Structure: KPI row -> sector donut + status/sentiment breakdown -> exposure
alerts table -> per-holding drilldown (expanders). Dashboard-glance-first,
detail-on-demand, in the spirit of the Power BI-style reference - but every
number on the page comes straight from orchestrator.run()'s real output,
nothing is a placeholder.

Usage:
    streamlit run app.py
"""

import json
import os
import tempfile
from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from src.agents.portfolio_health_agent import load_portfolio
from src import orchestrator

st.set_page_config(page_title="Agentic Portfolio Guardian", page_icon="🛡️", layout="wide")

THESES_PATH = "data/theses.json"
DEFAULT_PORTFOLIO_PATH = "data/sample_portfolio.csv"

# ---------------- Palette ----------------
COLORS = {
    "bg": "#120B0D",
    "bg_card": "#1E1417",
    "border": "#3A262B",
    "text": "#F3E9E4",
    "text_muted": "#B29A9D",
    "accent": "#8C2F39",
    "broken": "#C1444F",     # BROKEN / HIGH severity / negative sentiment
    "weakening": "#C9973B",  # WEAKENING / MEDIUM severity
    "holds": "#7C9473",      # HOLDS / LOW severity / positive sentiment
    "neutral": "#8FA3B0",    # N/A / no data / neutral sentiment
}
DONUT_SEQUENCE = ["#8C2F39", "#C9973B", "#7C9473", "#8FA3B0", "#C98A93", "#5C4A4E", "#B0855F", "#4E6E63"]

STATUS_KIND = {"HOLDS": "holds", "WEAKENING": "weakening", "BROKEN": "broken"}
SEVERITY_KIND = {"HIGH": "broken", "MEDIUM": "weakening", "LOW": "holds"}
SENTIMENT_KIND = {"positive": "holds", "neutral": "neutral", "negative": "broken"}


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {COLORS['text']};
    }}
    h1, h2, h3 {{
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #170D10;
        border-right: 1px solid {COLORS['border']};
    }}
    [data-testid="stExpander"] {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
    }}

    .pipeline {{ display: flex; align-items: center; margin: 0.5rem 0 1.75rem 0; flex-wrap: wrap; }}
    .pipeline .step {{
        font-size: 0.82rem; font-weight: 500; color: {COLORS['text_muted']};
        background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};
        border-radius: 999px; padding: 0.3rem 0.85rem; white-space: nowrap;
    }}
    .pipeline .arrow {{ color: {COLORS['accent']}; margin: 0 0.4rem; font-size: 0.9rem; }}

    /* ---- KPI cards ---- */
    .kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
    .kpi-card {{
        flex: 1; min-width: 170px; background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']}; border-radius: 10px; padding: 1rem 1.2rem;
    }}
    .kpi-card .label {{ font-size: 0.76rem; color: {COLORS['text_muted']}; margin-bottom: 0.35rem; }}
    .kpi-card .value {{ font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 600; line-height: 1.1; }}
    .kpi-card .delta {{ font-size: 0.78rem; color: {COLORS['text_muted']}; margin-top: 0.3rem; }}
    .progress-track {{ background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; margin-top: 0.6rem; overflow: hidden; }}
    .progress-fill {{ height: 100%; border-radius: 999px; }}

    /* ---- panel wrapper (for donut / breakdowns) ---- */
    .panel {{
        background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};
        border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 1rem; height: 100%;
    }}
    .panel .panel-title {{ font-size: 0.88rem; font-weight: 600; color: {COLORS['text']}; margin-bottom: 0.9rem; }}

    /* ---- stacked breakdown bars ---- */
    .stack-bar {{ display: flex; height: 10px; border-radius: 999px; overflow: hidden; margin-bottom: 0.7rem; background: rgba(255,255,255,0.05); }}
    .stack-bar .segment {{ height: 100%; }}
    .stack-bar .segment.holds {{ background: {COLORS['holds']}; }}
    .stack-bar .segment.weakening {{ background: {COLORS['weakening']}; }}
    .stack-bar .segment.broken {{ background: {COLORS['broken']}; }}
    .stack-bar .segment.neutral {{ background: {COLORS['neutral']}; }}
    .legend-row {{ font-size: 0.8rem; color: {COLORS['text_muted']}; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.4rem; }}
    .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
    .legend-dot.holds {{ background: {COLORS['holds']}; }}
    .legend-dot.weakening {{ background: {COLORS['weakening']}; }}
    .legend-dot.broken {{ background: {COLORS['broken']}; }}
    .legend-dot.neutral {{ background: {COLORS['neutral']}; }}

    /* ---- badges ---- */
    .badge {{ display: inline-block; font-size: 0.85rem; font-weight: 500; padding: 0.3rem 0.8rem; border-radius: 6px; margin-bottom: 0.6rem; }}
    .badge.holds     {{ background: rgba(124,148,115,0.16); color: {COLORS['holds']}; border: 1px solid rgba(124,148,115,0.4); }}
    .badge.weakening {{ background: rgba(201,151,59,0.16);  color: {COLORS['weakening']}; border: 1px solid rgba(201,151,59,0.4); }}
    .badge.broken    {{ background: rgba(193,68,79,0.18);   color: {COLORS['broken']}; border: 1px solid rgba(193,68,79,0.45); }}
    .badge.neutral   {{ background: rgba(143,163,176,0.16); color: {COLORS['neutral']}; border: 1px solid rgba(143,163,176,0.4); }}

    /* ---- alert banners ---- */
    .alert-box {{ border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 1rem; border-left: 3px solid; font-size: 0.92rem; }}
    .alert-box.holds     {{ background: rgba(124,148,115,0.10); border-color: {COLORS['holds']}; }}
    .alert-box.weakening {{ background: rgba(201,151,59,0.10);  border-color: {COLORS['weakening']}; }}
    .alert-box.broken    {{ background: rgba(193,68,79,0.12);   border-color: {COLORS['broken']}; }}
    .alert-box.neutral   {{ background: rgba(143,163,176,0.10); border-color: {COLORS['neutral']}; }}

    /* ---- exposure table ---- */
    .exp-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    .exp-table th {{ text-align: left; color: {COLORS['text_muted']}; font-weight: 500; padding: 0.5rem 0.7rem; border-bottom: 1px solid {COLORS['border']}; }}
    .exp-table td {{ padding: 0.55rem 0.7rem; border-bottom: 1px solid {COLORS['border']}; }}
    </style>
    """, unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{text}</span>'


def alert_box(html_content: str, kind: str):
    st.markdown(f'<div class="alert-box {kind}">{html_content}</div>', unsafe_allow_html=True)


def score_kind(score: float) -> str:
    if score is None:
        return "neutral"
    if score >= 70:
        return "holds"
    if score >= 40:
        return "weakening"
    return "broken"


def kpi_card(label: str, value: str, delta: str = None, progress_pct: float = None, progress_kind: str = "holds") -> str:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    progress_html = ""
    if progress_pct is not None:
        pct = max(0, min(100, progress_pct))
        progress_html = f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%; background:{COLORS[progress_kind]};"></div></div>'
    return f'''<div class="kpi-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}{progress_html}
    </div>'''


def pipeline_tracker(stages: list):
    steps_html = ""
    for i, stage in enumerate(stages):
        steps_html += f'<span class="step">{stage}</span>'
        if i < len(stages) - 1:
            steps_html += '<span class="arrow">&rarr;</span>'
    st.markdown(f'<div class="pipeline">{steps_html}</div>', unsafe_allow_html=True)


def breakdown_panel(title: str, counts: Counter, kind_map: dict, default_kind="neutral"):
    total = sum(counts.values()) or 1
    segments_html = "".join(
        f'<div class="segment {kind_map.get(label, default_kind)}" style="width:{(n / total) * 100:.1f}%;"></div>'
        for label, n in counts.items()
    )
    legend_html = "".join(
        f'<div class="legend-row"><span class="legend-dot {kind_map.get(label, default_kind)}"></span>{label.capitalize()}: {n}</div>'
        for label, n in counts.items()
    )
    st.markdown(f'''<div class="panel">
        <div class="panel-title">{title}</div>
        <div class="stack-bar">{segments_html}</div>
        {legend_html}
    </div>''', unsafe_allow_html=True)


def sector_donut(sector_weights: dict):
    labels = list(sector_weights.keys())
    values = list(sector_weights.values())
    colors = [DONUT_SEQUENCE[i % len(DONUT_SEQUENCE)] for i in range(len(labels))]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color=COLORS["bg_card"], width=2)),
        textinfo="label+percent", textfont=dict(color=COLORS["text"], size=12),
    )])
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        font=dict(family="IBM Plex Sans", color=COLORS["text"]),
    )
    return fig


def exposure_table_html(alerts: list) -> str:
    rows = ""
    for a in alerts:
        label = a.get("ticker") or a.get("sector") or a.get("theme")
        rows += (
            f"<tr><td>{a.get('type')}</td><td>{label}</td>"
            f"<td>{(a.get('weight') or 0) * 100:.1f}%</td>"
            f"<td>{(a.get('limit') or 0) * 100:.0f}%</td></tr>"
        )
    return f'''<table class="exp-table">
        <tr><th>Type</th><th>Item</th><th>Weight</th><th>Limit</th></tr>
        {rows}
    </table>'''


def load_theses_file(path: str = THESES_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_theses_file(theses: dict, path: str = THESES_PATH):
    with open(path, "w") as f:
        json.dump(theses, f, indent=2)


inject_css()

# ---------------- Sidebar ----------------

st.sidebar.header("Portfolio Guardian")

st.sidebar.subheader("1. Portfolio")
portfolio_path_input = st.sidebar.text_input(
    "Portfolio CSV path", value=DEFAULT_PORTFOLIO_PATH,
    help="Path on disk to a portfolio CSV (same format as sample_portfolio.csv)",
)
uploaded_file = st.sidebar.file_uploader("...or upload a CSV", type=["csv"])

active_portfolio_path = portfolio_path_input
if uploaded_file is not None:
    tmp_csv_path = os.path.join(tempfile.gettempdir(), "guardian_uploaded_portfolio.csv")
    with open(tmp_csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    active_portfolio_path = tmp_csv_path
    st.sidebar.caption(f"Using uploaded file (saved to {tmp_csv_path})")

holdings = []
load_error = None
try:
    holdings = load_portfolio(active_portfolio_path)
except Exception as e:
    load_error = str(e)

if load_error:
    st.sidebar.error(f"Could not load portfolio: {load_error}")

st.sidebar.subheader("2. Investment theses")
st.sidebar.caption(
    "Pre-filled from data/theses.json. Edits here are saved back to that "
    "file when you click Run, so thesis_node picks them up - no manual "
    "JSON editing needed."
)

all_theses = load_theses_file()
edited_theses = dict(all_theses)

for h in holdings:
    ticker = h["ticker"]
    edited_theses[ticker] = st.sidebar.text_input(
        ticker, value=all_theses.get(ticker, ""), key=f"thesis_{ticker}"
    )

st.sidebar.subheader("3. Options")
fetch_prices = st.sidebar.checkbox("Fetch live prices (yfinance history)", value=True)
fetch_market = st.sidebar.checkbox("Fetch market metadata (industry/cap/volume)", value=True)

run_clicked = st.sidebar.button("Run Full Analysis", type="primary", disabled=not holdings)


# ---------------- Main area ----------------

st.title("Agentic Portfolio Guardian")
pipeline_tracker(["Portfolio Health", "Thesis", "Market Intelligence", "Early Warning", "Merge"])

if run_clicked:
    save_theses_file(edited_theses)
    with st.spinner(
        "Running the full pipeline - this calls Ollama per holding across "
        "3 agents, so a full portfolio can take a few minutes ..."
    ):
        final_state = orchestrator.run(
            active_portfolio_path,
            limit=None,
            fetch_prices=fetch_prices,
            fetch_market=fetch_market,
        )
    st.session_state["final_state"] = final_state

final_state = st.session_state.get("final_state")

if final_state is None:
    alert_box("Set your portfolio + theses in the sidebar, then click <b>Run Full Analysis</b>.", "neutral")
else:
    errors = final_state.get("errors", [])
    if errors:
        error_lines = "<br>".join(
            f"<b>[{e.get('node')}]</b> {e.get('ticker') or '(portfolio-level)'}: {e.get('error')}"
            for e in errors
        )
        alert_box(f"<b>{len(errors)} node(s) reported an error</b> - showing partial results below.<br>{error_lines}", "broken")

    final_output = final_state.get("final_output") or {}
    portfolio_health = final_output.get("portfolio_health")
    per_holding = final_output.get("per_holding") or []

    # ---- roll-ups for the dashboard row (computed once, used by KPIs + panels) ----
    status_counts = Counter(h.get("thesis_status") or "Unknown" for h in per_holding)
    sentiment_counts = Counter(h.get("market_sentiment") or "unknown" for h in per_holding)
    flagged_holdings = [h for h in per_holding if h.get("redflag_alerts")]
    status_kind_map = {**STATUS_KIND, "Unknown": "neutral"}
    sentiment_kind_map = {**SENTIMENT_KIND, "unknown": "neutral"}

    # ---- KPI row ----
    if portfolio_health:
        largest_holding = portfolio_health.get("largest_holding") or {}
        largest_sector = portfolio_health.get("largest_sector") or {}
        health_score = portfolio_health.get("portfolio_health_score")
        div_score = (portfolio_health.get("diversification_metrics") or {}).get("diversification_score")

        cards = '<div class="kpi-row">'
        cards += kpi_card("Health Score", f"{health_score if health_score is not None else 'N/A'}",
                           progress_pct=health_score, progress_kind=score_kind(health_score))
        cards += kpi_card("Diversification Score", f"{div_score if div_score is not None else 'N/A'}",
                           progress_pct=div_score, progress_kind=score_kind(div_score))
        cards += kpi_card("Largest Holding", largest_holding.get("ticker", "N/A"),
                           f"{(largest_holding.get('weight') or 0) * 100:.1f}% of portfolio")
        cards += kpi_card("Largest Sector", largest_sector.get("sector", "N/A"),
                           f"{(largest_sector.get('weight') or 0) * 100:.1f}% of portfolio")
        cards += kpi_card("Holdings Flagged", str(len(flagged_holdings)),
                           f"of {len(per_holding)} holdings")
        cards += '</div>'
        st.markdown(cards, unsafe_allow_html=True)
    else:
        alert_box("Portfolio health data unavailable (see errors above).", "neutral")

    # ---- donut + breakdown panels ----
    if portfolio_health or per_holding:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown('<div class="panel"><div class="panel-title">Sector Allocation</div>', unsafe_allow_html=True)
            sector_weights = (portfolio_health or {}).get("sector_weights") or {}
            if sector_weights:
                st.plotly_chart(sector_donut(sector_weights), use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown("_No sector data available._")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            if per_holding:
                breakdown_panel("Thesis Status", status_counts, status_kind_map)
                breakdown_panel("Market Sentiment", sentiment_counts, sentiment_kind_map)

    # ---- exposure alerts ----
    if portfolio_health:
        exposure_alerts = portfolio_health.get("exposure_alerts") or []
        st.subheader("Exposure Alerts")
        if exposure_alerts:
            st.markdown(exposure_table_html(exposure_alerts), unsafe_allow_html=True)
        else:
            alert_box("No exposure alerts.", "holds")

    # ---- per-holding drilldown ----
    st.header("Per-Holding Detail")
    for holding in per_holding:
        ticker = holding.get("ticker")
        status = holding.get("thesis_status")
        with st.expander(f"{ticker} — thesis: {status or 'N/A'}"):
            status_kind = STATUS_KIND.get(status, "neutral")
            st.markdown(badge(f"Thesis status: {status or 'N/A'}", status_kind), unsafe_allow_html=True)
            st.markdown(f"**Reasoning:** {holding.get('thesis_reasoning') or '_No reasoning available._'}")

            st.markdown("---")
            st.markdown(f"**Market sentiment:** {holding.get('market_sentiment') or 'N/A'}")
            st.markdown(f"**Market summary:** {holding.get('market_summary') or '_No summary available._'}")

            st.markdown("---")
            alerts = holding.get("redflag_alerts") or []
            if alerts:
                st.markdown(f"**Red-flag alerts ({len(alerts)}):**")
                for alert in alerts:
                    sev = alert.get("severity")
                    sev_kind = SEVERITY_KIND.get(sev, "neutral")
                    alert_box(
                        f"<b>[{alert.get('flag_type') or 'flag'}]</b> {alert.get('headline')}"
                        f"<div style='margin-top:0.3rem; color:{COLORS['text_muted']}; font-size:0.85rem;'>"
                        f"{alert.get('reasoning') or ''}</div>",
                        sev_kind,
                    )
            else:
                st.markdown("**Red-flag alerts:** none")