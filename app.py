"""
Agentic Portfolio Guardian - Streamlit app (design-system rebuild)

Visual language: dark wine/burgundy atmosphere, translucent smoked-glass
panels, muted champagne gold accents, restrained crimson for alerts, clean
sans-serif (Inter) throughout. Dashboard page is single-screen/no-scroll;
Thesis Monitor is the scrollable detail page. Other nav items are
placeholders until we build them out.

Usage:
    streamlit run app.py
"""

import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime

import streamlit as st

from src.agents.portfolio_health_agent import load_portfolio
from src import orchestrator

st.set_page_config(page_title="Portfolio Intelligence", page_icon="🍷", layout="wide")

THESES_PATH = "data/theses.json"
DEFAULT_PORTFOLIO_PATH = "data/sample_portfolio.csv"

# ================= Design tokens (muted palette, flat — no neon glow) =================
C = {
    "gold": "#C9A567",
    "gold_light": "#DFC48A",
    "gold_dark": "#7A5C30",
    "gold_alt": "#B89355",
    "red_light": "#C05A62",
    "red_mid": "#A83C46",
    "red_deep": "#6E1F28",
    "red_darkest": "#451318",
    "text": "#EDE3D8",
    "text_muted": "#A6968A",
    "green": "#8FA37A",
    "glass_bg": "rgba(35,20,22,0.55)",
    "glass_border": "rgba(180,150,100,0.15)",
}

STATUS_COLOR = {"HOLDS": C["green"], "WEAKENING": C["gold"], "BROKEN": C["red_mid"], None: C["text_muted"]}
SENTIMENT_COLOR = {"positive": C["green"], "neutral": C["text_muted"], "negative": C["red_mid"]}

NAV_ITEMS = ["Dashboard", "Portfolio", "Risk", "Thesis", "Markets", "Early Warning", "Invest", "Reports", "Settings"]
BUILT_PAGES = {"Dashboard", "Thesis", "Settings"}

st.session_state.setdefault("page", "Dashboard")
st.session_state.setdefault("final_state", None)
st.session_state.setdefault("last_run_ts", None)


def inject_css(no_scroll: bool):
    scroll_lock = """
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1.1rem !important; padding-bottom: 0.5rem !important; }
    html, body, [data-testid="stAppViewContainer"] { height: 100vh !important; overflow: hidden !important; }
    section[data-testid="stSidebar"] { overflow: hidden !important; }
    """ if no_scroll else """
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1.1rem !important; }
    """

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {C['text']}; }}
    h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; font-weight: 600; margin: 0; letter-spacing: -0.01em; }}

    /* ---- converging-ray dark background (matches reference art) ---- */
    .stApp {{
        background:
          repeating-conic-gradient(from 0deg at 22% 15%,
            rgba(198,56,70,0.10) 0deg 4deg, rgba(0,0,0,0) 4deg 14deg),
          radial-gradient(circle at 22% 15%, rgba(138,32,43,0.35), transparent 55%),
          radial-gradient(circle at 90% 85%, rgba(53,20,23,0.25), transparent 50%),
          #0A0506;
        background-blend-mode: screen, normal, normal;
    }}
    {scroll_lock}

    /* ---- floating glass sidebar ---- */
    section[data-testid="stSidebar"] > div:first-child {{
        margin: 1rem 0 1rem 1rem; border-radius: 18px;
        background: rgba(30,17,19,0.55); backdrop-filter: blur(20px);
        border: 1px solid rgba(180,150,100,0.12);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        padding: 0.5rem;
    }}
    section[data-testid="stSidebar"] button {{
        background: transparent !important; border: 1px solid transparent !important;
        color: {C['text_muted']} !important; text-align: left !important; font-weight: 500 !important;
        border-radius: 10px !important; transition: all 0.15s ease !important;
    }}
    section[data-testid="stSidebar"] button:hover {{
        background: rgba(255,255,255,0.04) !important; border-color: rgba(180,150,100,0.2) !important;
        color: {C['text']} !important;
    }}

    .nav-active {{
        display: flex; align-items: center; gap: 0.5rem;
        background: rgba(201,165,103,0.12); border: 1px solid rgba(201,165,103,0.3);
        color: {C['gold_light']} !important; border-radius: 10px; padding: 0.5rem 0.9rem;
        font-weight: 600; font-size: 0.92rem; margin-bottom: 0.15rem;
    }}

    /* ---- glass panel base ---- */
    .glass {{
        position: relative; overflow: hidden;
        background: {C['glass_bg']};
        backdrop-filter: blur(24px) saturate(140%);
        -webkit-backdrop-filter: blur(24px) saturate(140%);
        border: 1px solid {C['glass_border']};
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .glass::before {{
        content: ""; position: absolute; inset: -40% -40% auto -40%; height: 180%;
        background: linear-gradient(115deg, transparent 25%, rgba(255,255,255,0.03) 45%, transparent 60%);
        pointer-events: none;
    }}
    .glass:hover {{ border-color: rgba(180,150,100,0.3); box-shadow: 0 10px 32px rgba(0,0,0,0.4); }}

    /* ---- header ---- */
    .app-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.1rem; flex-wrap: wrap; gap: 0.8rem; }}
    .app-header .kicker {{ font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: {C['text_muted']}; }}
    .app-header h1 {{ font-size: 1.35rem; margin-top: 0.1rem; }}
    .app-header .meta {{ font-size: 0.76rem; color: {C['text_muted']}; margin-top: 0.15rem; }}
    .header-right {{ display: flex; align-items: center; gap: 0.7rem; }}
    .pill-btn {{
        font-size: 0.78rem; color: {C['text_muted']}; background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 999px; padding: 0.42rem 0.9rem;
    }}
    .icon-dot {{ width: 34px; height: 34px; border-radius: 50%; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); display: flex; align-items: center; justify-content: center; color: {C['text_muted']}; }}
    .avatar {{ width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, {C['gold_light']}, {C['gold_dark']}); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #241505; }}

    /* flat button - no neon glow */
    div[data-testid="stButton"] button[kind="primary"] {{
        background: linear-gradient(135deg, {C['red_light']}, {C['red_deep']}) !important;
        border: 1px solid rgba(192,90,98,0.5) !important;
        border-radius: 999px !important; font-weight: 600 !important; font-size: 0.82rem !important;
        padding: 0.4rem 1.1rem !important;
    }}

    /* ---- kickers / labels ---- */
    .section-kicker {{ font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase; color: {C['text_muted']}; margin-bottom: 0.5rem; }}

    /* ---- badges ---- */
    .badge {{ display: inline-block; font-size: 0.76rem; font-weight: 600; padding: 0.24rem 0.7rem; border-radius: 999px; }}
    .badge.holds     {{ background: rgba(143,163,122,0.14); color: {C['green']}; border: 1px solid rgba(143,163,122,0.35); }}
    .badge.weakening {{ background: rgba(201,165,103,0.14); color: {C['gold']}; border: 1px solid rgba(201,165,103,0.35); }}
    .badge.broken    {{ background: rgba(168,60,70,0.16); color: {C['red_light']}; border: 1px solid rgba(168,60,70,0.4); }}
    .badge.neutral   {{ background: rgba(255,255,255,0.05); color: {C['text_muted']}; border: 1px solid rgba(255,255,255,0.1); }}

    .progress-track {{ background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden; }}
    .progress-fill {{ height: 100%; border-radius: 999px; }}

    /* ---- table ---- */
    table.premium-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    table.premium-table th {{ text-align: left; color: {C['text_muted']}; font-weight: 500; padding: 0.45rem 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    table.premium-table td {{ padding: 0.55rem 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    table.premium-table tr:hover td {{ background: rgba(201,165,103,0.04); }}
    table.premium-table tr:last-child td {{ border-bottom: none; }}

    /* ---- expander (thesis monitor) ---- */
    [data-testid="stExpander"] {{
        background: {C['glass_bg']}; border: 1px solid {C['glass_border']}; border-radius: 14px;
        backdrop-filter: blur(20px);
    }}

    .scroll-box {{ overflow-y: auto; }}
    .scroll-box::-webkit-scrollbar {{ width: 6px; }}
    .scroll-box::-webkit-scrollbar-thumb {{ background: rgba(201,165,103,0.28); border-radius: 999px; }}
    </style>
    """, unsafe_allow_html=True)


# ================= Helpers =================

def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{text}</span>'

def status_kind(status):
    return {"HOLDS": "holds", "WEAKENING": "weakening", "BROKEN": "broken"}.get(status, "neutral")

def severity_kind(sev):
    return {"HIGH": "broken", "MEDIUM": "weakening", "LOW": "holds"}.get(sev, "neutral")

def load_theses_file(path: str = THESES_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def save_theses_file(theses: dict, path: str = THESES_PATH):
    with open(path, "w") as f:
        json.dump(theses, f, indent=2)


def custom_spinner_html() -> str:
    """Rotating glowing-arc SVG (SMIL-animated, runs client-side so it
    keeps spinning during the blocking orchestrator.run() call below)."""
    return f'''
    <div style="display:flex; flex-direction:column; align-items:center; padding:2.2rem 0; gap:0.9rem;">
      <svg width="70" height="70" viewBox="0 0 70 70">
        <defs>
          <linearGradient id="arcGlow" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{C['red_deep']}" stop-opacity="0"/>
            <stop offset="55%" stop-color="{C['red_mid']}" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="{C['red_light']}" stop-opacity="1"/>
          </linearGradient>
        </defs>
        <circle cx="35" cy="35" r="28" fill="none" stroke="url(#arcGlow)" stroke-width="3"
                stroke-linecap="round" stroke-dasharray="140 200">
          <animateTransform attributeName="transform" type="rotate" from="0 35 35" to="360 35 35"
                             dur="1.1s" repeatCount="indefinite"/>
        </circle>
      </svg>
      <span style="font-size:0.78rem; color:{C['text_muted']};">
        Running the full pipeline — Ollama calls per holding across 3 agents, this can take a few minutes...
      </span>
    </div>'''


def run_analysis():
    theses_now = st.session_state.get("theses_edit") or {}
    if theses_now:
        save_theses_file(theses_now)
    portfolio_path = st.session_state.get("portfolio_path", DEFAULT_PORTFOLIO_PATH)
    fetch_prices = st.session_state.get("fetch_prices", True)
    fetch_market = st.session_state.get("fetch_market", True)

    placeholder = st.empty()
    placeholder.markdown(custom_spinner_html(), unsafe_allow_html=True)
    final_state = orchestrator.run(portfolio_path, limit=None, fetch_prices=fetch_prices, fetch_market=fetch_market)
    placeholder.empty()

    st.session_state["final_state"] = final_state
    st.session_state["last_run_ts"] = datetime.now().strftime("%b %d, %Y · %H:%M")


def svg_donut(sector_weights: dict, size: int = 130, stroke: int = 16) -> str:
    """Hand-rolled ring chart as a single inline SVG string, so it can be
    embedded directly inside a combined markdown call (Plotly can't -
    it's a native widget that renders as its own separate block, which
    is exactly the split-across-calls problem the glass boxes had)."""
    labels = list(sector_weights.keys())
    values = list(sector_weights.values())
    total = sum(values) or 1
    palette = [C["red_mid"], C["gold"], C["green"], C["text_muted"], C["gold_alt"], C["red_deep"]]
    cx = cy = size / 2
    r = size / 2 - stroke / 2 - 2
    circumference = 2 * math.pi * r
    offset = 0.0
    arcs = ""
    for i, v in enumerate(values):
        frac = v / total
        dash = frac * circumference
        gap = circumference - dash
        color = palette[i % len(palette)]
        arcs += (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-dasharray="{dash:.2f} {gap:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 {cx} {cy})" '
            f'stroke-linecap="butt" />'
        )
        offset += dash
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{arcs}</svg>'


# ================= Sidebar (nav) =================

def render_sidebar():
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.6rem 0.6rem 1rem;">'
        f'<div style="width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,{C["red_light"]},{C["red_deep"]});"></div>'
        f'<span style="font-weight:600;font-size:0.92rem;">Portfolio Guardian</span></div>',
        unsafe_allow_html=True,
    )
    for item in NAV_ITEMS:
        if item == st.session_state["page"]:
            st.sidebar.markdown(f'<div class="nav-active">{item}</div>', unsafe_allow_html=True)
        else:
            if st.sidebar.button(item, key=f"nav_{item}", use_container_width=True):
                st.session_state["page"] = item
                st.rerun()


# ================= Header =================

def render_header():
    last_run = st.session_state.get("last_run_ts") or "Not run yet"
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f'''
        <div class="app-header">
          <div>
            <div class="kicker">Portfolio Intelligence</div>
            <h1>Chahat Saini</h1>
            <div class="meta">Last analysis · {last_run}</div>
          </div>
        </div>''', unsafe_allow_html=True)
    with col2:
        b1, b2, b3, b4 = st.columns([1.6, 0.6, 0.6, 0.6])
        with b1:
            if st.button("Run Full Analysis", type="primary", use_container_width=True):
                run_analysis()
                st.rerun()
        with b2:
            st.markdown('<div class="icon-dot">🔍</div>', unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="icon-dot">🔔</div>', unsafe_allow_html=True)
        with b4:
            st.markdown('<div class="avatar">CS</div>', unsafe_allow_html=True)


# ================= Pages =================

def page_settings():
    st.markdown('<div class="section-kicker">Data Settings</div>', unsafe_allow_html=True)

    portfolio_path_input = st.text_input(
        "Portfolio CSV path", value=st.session_state.get("portfolio_path", DEFAULT_PORTFOLIO_PATH),
        help="Path on disk to a portfolio CSV (same format as sample_portfolio.csv)",
    )
    uploaded_file = st.file_uploader("...or upload a CSV", type=["csv"])

    active_path = portfolio_path_input
    if uploaded_file is not None:
        tmp_csv_path = os.path.join(tempfile.gettempdir(), "guardian_uploaded_portfolio.csv")
        with open(tmp_csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_path = tmp_csv_path
        st.caption(f"Using uploaded file (saved to {tmp_csv_path})")

    st.session_state["portfolio_path"] = active_path

    holdings = []
    try:
        holdings = load_portfolio(active_path)
    except Exception as e:
        st.error(f"Could not load portfolio: {e}")
    st.session_state["holdings_cache"] = holdings

    st.markdown("---")
    st.markdown('<div class="section-kicker">Investment Theses</div>', unsafe_allow_html=True)
    all_theses = load_theses_file()
    edited = dict(all_theses)
    for h in holdings:
        ticker = h["ticker"]
        edited[ticker] = st.text_input(ticker, value=all_theses.get(ticker, ""), key=f"thesis_{ticker}")
    st.session_state["theses_edit"] = edited

    st.markdown("---")
    st.markdown('<div class="section-kicker">Analysis Preferences</div>', unsafe_allow_html=True)
    st.session_state["fetch_prices"] = st.checkbox("Fetch live prices (yfinance history)", value=st.session_state.get("fetch_prices", True))
    st.session_state["fetch_market"] = st.checkbox("Fetch market metadata (industry/cap/volume)", value=st.session_state.get("fetch_market", True))

    st.markdown("---")
    if st.button("Run Full Analysis", type="primary"):
        run_analysis()
        st.rerun()


def page_placeholder(name: str):
    st.markdown(
        f'<div class="glass" style="padding:3rem 2rem; text-align:center;">'
        f'<div class="section-kicker">{name}</div>'
        f'<h2 style="color:{C["text_muted"]}; font-weight:500;">Not built yet</h2>'
        f'<p style="color:{C["text_muted"]}; font-size:0.85rem;">This view is part of the design system but isn\'t wired up yet. Ask to build the {name} page next.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def page_dashboard():
    final_state = st.session_state.get("final_state")
    if not final_state:
        st.markdown(
            f'<div class="glass" style="padding:2.5rem; text-align:center;">'
            f'<h2 style="color:{C["text_muted"]}; font-weight:500; margin-bottom:0.6rem;">No analysis yet</h2>'
            f'<p style="color:{C["text_muted"]}; margin:0;">Go to Settings to set your portfolio and theses, then click Run Full Analysis.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    errors = final_state.get("errors", [])
    final_output = final_state.get("final_output") or {}
    ph = final_output.get("portfolio_health") or {}
    per_holding = final_output.get("per_holding") or []

    status_counts = Counter(h.get("thesis_status") or "Unknown" for h in per_holding)
    sentiment_counts = Counter(h.get("market_sentiment") or "unknown" for h in per_holding)
    flagged = [h for h in per_holding if h.get("redflag_alerts")]
    exposure_alerts = ph.get("exposure_alerts") or []
    health_score = ph.get("portfolio_health_score")
    div_score = (ph.get("diversification_metrics") or {}).get("diversification_score")
    largest_holding = ph.get("largest_holding") or {}
    largest_sector = ph.get("largest_sector") or {}
    sector_weights = ph.get("sector_weights") or {}
    perf = ph.get("performance_metrics") or {}
    risk = ph.get("risk_metrics") or {}

    n_broken = status_counts.get("BROKEN", 0)
    n_weak = status_counts.get("WEAKENING", 0)
    synthesis = (
        f"{n_broken} thesis broken and {n_weak} weakening out of {len(per_holding)} holdings. "
        + (f"{len(flagged)} holding(s) carry an active red-flag alert. " if flagged else "No red-flag alerts on any holding. ")
        + (f"Largest exposure concern: {exposure_alerts[0].get('type')} on "
           f"{exposure_alerts[0].get('ticker') or exposure_alerts[0].get('sector')}."
           if exposure_alerts else "No exposure-limit breaches.")
    )

    if errors:
        st.markdown(
            f'<div class="glass" style="padding:0.8rem 1.2rem; border-color: rgba(168,60,70,0.4); margin-bottom:0.7rem;">'
            f'<span style="color:{C["red_light"]}; font-size:0.85rem;"><b>{len(errors)} node error(s)</b> — showing partial results.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- TOP ALERT STRIP (full-width, matches reference design) ----
    if exposure_alerts or flagged:
        chips = []
        for a in exposure_alerts[:3]:
            label = a.get("ticker") or a.get("sector") or a.get("theme")
            chips.append(f'{badge(a.get("type"), "weakening")} {label} {(a.get("weight") or 0)*100:.0f}%')
        for h in flagged[:2]:
            chips.append(f'{badge("red flag", "broken")} {h.get("ticker")}')
        chips_html = " &nbsp;&nbsp; ".join(chips)
        total_alerts = len(exposure_alerts) + len(flagged)
        st.markdown(
            f'<div class="glass" style="padding:0.6rem 1.2rem; margin-bottom:0.9rem; '
            f'display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">'
            f'<span style="font-size:0.82rem; color:{C["gold_light"]};">'
            f'⚠ <b>{total_alerts} ALERT{"S" if total_alerts != 1 else ""}</b> &nbsp; {chips_html}</span>'
            f'<span style="font-size:0.78rem; color:{C["text_muted"]};">View all →</span></div>',
            unsafe_allow_html=True,
        )

    # ---- HERO ROW: Portfolio Health + AI Synthesis ----
    hero_cols = st.columns([1, 1.8])
    with hero_cols[0]:
        pct = max(0, min(100, health_score or 0))
        st.markdown(
            f'<div class="glass" style="padding:1.2rem 1.4rem; height:100%;">'
            f'<div class="section-kicker">Portfolio Health</div>'
            f'<div style="font-size:2.1rem; font-weight:700; color:{C["gold_light"]};">{health_score if health_score is not None else "N/A"}</div>'
            f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%; background:linear-gradient(90deg,{C["gold_dark"]},{C["gold_light"]});"></div></div>'
            f'<div style="font-size:0.75rem; color:{C["text_muted"]}; margin-top:0.5rem;">Diversification: {div_score if div_score is not None else "N/A"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with hero_cols[1]:
        st.markdown(
            f'<div class="glass" style="padding:1.2rem 1.4rem; height:100%;">'
            f'<div class="section-kicker">AI Synthesis</div>'
            f'<p style="font-size:0.86rem; line-height:1.55; color:{C["text"]}; margin:0;">{synthesis}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    # ---- ROW 2: Performance & Risk / Allocation / Diversification ----
    row2 = st.columns([1.3, 1, 1])
    with row2[0]:
        metrics = [
            ("CAGR", perf.get("cagr")), ("Sharpe", perf.get("sharpe_ratio")),
            ("Volatility", risk.get("volatility_annualized")), ("Max DD", risk.get("max_drawdown")),
        ]
        metrics_html = '<div style="display:flex; gap:1.5rem; margin-top:0.4rem; flex-wrap:wrap;">'
        for label, val in metrics:
            display = f"{val:.2%}" if isinstance(val, float) and abs(val) < 5 else (f"{val}" if val is not None else "N/A")
            metrics_html += (
                f'<div><div style="font-size:0.68rem; color:{C["text_muted"]};">{label}</div>'
                f'<div style="font-size:1.05rem; font-weight:600; color:{C["gold_light"]};">{display}</div></div>'
            )
        metrics_html += '</div>'
        empty_note = (
            f'<p style="font-size:0.75rem; color:{C["text_muted"]}; margin-top:0.6rem;">'
            f'Run with price history enabled to populate these.</p>'
        ) if not perf and not risk else ''
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Performance &amp; Risk</div>{metrics_html}{empty_note}</div>',
            unsafe_allow_html=True,
        )

    with row2[1]:
        if sector_weights:
            body = (
                f'<div style="display:flex; justify-content:center; margin:0.2rem 0 0.4rem;">{svg_donut(sector_weights)}</div>'
                f'<div style="font-size:0.78rem; color:{C["text_muted"]}; text-align:center;">'
                f'Largest: {largest_sector.get("sector","N/A")} · {(largest_sector.get("weight") or 0)*100:.0f}%</div>'
            )
        else:
            body = f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No sector data</span>'
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Allocation</div>{body}</div>',
            unsafe_allow_html=True,
        )

    with row2[2]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Diversification</div>'
            f'<div style="font-size:1.7rem; font-weight:700; color:{C["gold_light"]};">{div_score if div_score is not None else "N/A"}</div>'
            f'<div style="font-size:0.78rem; color:{C["text_muted"]}; margin-top:0.4rem;">Largest holding: {largest_holding.get("ticker","N/A")} · {(largest_holding.get("weight") or 0)*100:.0f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    # ---- ROW 3: Market Pulse / Thesis Summary (links out to full detail) ----
    n_holds = status_counts.get("HOLDS", 0)
    row3 = st.columns([1, 1.6])
    with row3[0]:
        total_sent = sum(sentiment_counts.values()) or 1
        bars_html = ""
        for label in ["positive", "neutral", "negative"]:
            n = sentiment_counts.get(label, 0)
            pct = (n / total_sent) * 100
            color = SENTIMENT_COLOR[label]
            bars_html += (
                f'<div style="margin-bottom:0.5rem;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{C["text_muted"]};">'
                f'<span>{label.capitalize()}</span><span>{n}</span></div>'
                f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%; background:{color};"></div></div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Market Pulse</div>{bars_html}</div>',
            unsafe_allow_html=True,
        )

    with row3[1]:
        summary_chips = (
            f'{badge(f"{n_broken} Broken", "broken")} &nbsp; '
            f'{badge(f"{n_weak} Weakening", "weakening")} &nbsp; '
            f'{badge(f"{n_holds} Holds", "holds")}'
        )
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem;">'
            f'<div class="section-kicker">Thesis Monitor</div>'
            f'<div style="margin:0.4rem 0 0.2rem;">{summary_chips}</div>'
            f'<p style="font-size:0.8rem; color:{C["text_muted"]}; margin:0.6rem 0 0;">'
            f'{len(per_holding)} holdings tracked — full reasoning, sentiment, and red-flag detail on the Thesis page.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("View full thesis monitor →", use_container_width=True, key="goto_thesis"):
            st.session_state["page"] = "Thesis"
            st.rerun()


def page_thesis():
    final_state = st.session_state.get("final_state")
    if not final_state:
        st.markdown(
            f'<div class="glass" style="padding:2.5rem; text-align:center;">'
            f'<p style="color:{C["text_muted"]}; margin:0;">No analysis yet — run it from Settings first.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    per_holding = (final_state.get("final_output") or {}).get("per_holding") or []
    st.markdown('<div class="section-kicker">Thesis Monitor — Full Detail</div>', unsafe_allow_html=True)

    for h in per_holding:
        ticker = h.get("ticker")
        status = h.get("thesis_status")
        with st.expander(f"{ticker} — {status or 'N/A'}"):
            st.markdown(badge(status or "N/A", status_kind(status)), unsafe_allow_html=True)
            st.markdown(f"**Reasoning:** {h.get('thesis_reasoning') or '_No reasoning available._'}")
            st.markdown("---")
            st.markdown(f"**Market sentiment:** {h.get('market_sentiment') or 'N/A'}")
            st.markdown(f"**Market summary:** {h.get('market_summary') or '_No summary available._'}")
            st.markdown("---")
            alerts = h.get("redflag_alerts") or []
            if alerts:
                for a in alerts:
                    sev = a.get("severity")
                    flag_label = f'[{a.get("flag_type") or "flag"}] {sev or ""}'
                    st.markdown(
                        f'<div class="glass" style="padding:0.7rem 0.9rem; margin-top:0.4rem; border-color:rgba(168,60,70,0.4);">'
                        f'{badge(flag_label, severity_kind(sev))}'
                        f'<div style="margin-top:0.4rem; font-size:0.85rem;">{a.get("headline")}</div>'
                        f'<div style="margin-top:0.3rem; font-size:0.8rem; color:{C["text_muted"]};">{a.get("reasoning") or ""}</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
            else:
                st.markdown("**Red-flag alerts:** none")


# ================= Main =================

page = st.session_state["page"]
inject_css(no_scroll=(page == "Dashboard"))
render_sidebar()
render_header()

if page == "Dashboard":
    page_dashboard()
elif page == "Thesis":
    page_thesis()
elif page == "Settings":
    page_settings()
else:
    page_placeholder(page)