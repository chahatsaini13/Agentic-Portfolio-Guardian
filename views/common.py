"""
Shared design tokens, stateless helpers, and app-level actions used by
every page module in views/. Kept separate from app.py so each page file
only imports what it needs, instead of everything living in one big file.
"""

import json
import math
import os
from datetime import datetime

import streamlit as st

from src.agents.portfolio_health_agent import load_portfolio
from src import orchestrator
from src.scheduler import start_scheduler_once
from src.cache_store import load_latest_state

THESES_PATH = "data/theses.json"
DEFAULT_PORTFOLIO_PATH = "data/sample_portfolio.csv"

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


def start_background_scheduler_once():
    """Starts the scheduler once per Streamlit session (cheap session_state
    guard here). src/scheduler.py's own module-level singleton is the real
    guard against duplicate schedulers across multiple sessions in the same
    process - this just avoids re-checking on every rerun."""
    if not st.session_state.get("scheduler_started"):
        start_scheduler_once()
        st.session_state["scheduler_started"] = True


def init_state_from_cache():
    """Called once at app startup (from app.py, right after the
    st.session_state.setdefault() calls) so the dashboard shows the last
    scheduled run immediately instead of being blank until someone clicks
    'Run Full Analysis'. Never overwrites a final_state that's already
    populated in this session (e.g. from a manual run earlier)."""
    if st.session_state.get("final_state") is not None:
        return
    cached = load_latest_state()
    if cached is None:
        return
    st.session_state["final_state"] = {
        "final_output": cached["final_output"],
        "errors": cached["errors"],
    }
    st.session_state["last_run_ts"] = cached["last_updated"] + " (cached)"


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

def generate_gold_shades(n: int) -> list:
    """Self-assigning luxe palette: n shades of one warm gold/amber hue,
    interpolated bright-to-deep-bronze, so any category count (however
    many holdings/sectors/industries exist) gets a cohesive but visibly
    distinct set of colors - no manual palette list to maintain."""
    if n <= 0:
        return []
    light = (240, 200, 117)   # bright glowing amber
    dark = (74, 51, 10)       # deep bronze
    shades = []
    for i in range(n):
        t = i / max(1, n - 1)  # 0 -> 1 across the n stops
        r = round(light[0] + (dark[0] - light[0]) * t)
        g = round(light[1] + (dark[1] - light[1]) * t)
        b = round(light[2] + (dark[2] - light[2]) * t)
        shades.append(f"#{r:02X}{g:02X}{b:02X}")
    return shades

def svg_donut(sector_weights: dict, size: int = 130, stroke: int = 16) -> str:
    """Hand-rolled ring chart as a single inline SVG string, so it can be
    embedded directly inside a combined markdown call (Plotly can't -
    it's a native widget that renders as its own separate block, which
    is exactly the split-across-calls problem the glass boxes had)."""
    labels = list(sector_weights.keys())
    values = list(sector_weights.values())
    total = sum(values) or 1
    palette = generate_gold_shades(len(values))
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


def donut_legend_html(sector_weights: dict) -> str:
    """Color-coded legend for svg_donut() - same palette/order so colors
    match exactly."""
    if not sector_weights:
        return ""
    palette = generate_gold_shades(len(sector_weights))
    items = ""
    for i, (label, weight) in enumerate(sector_weights.items()):
        color = palette[i % len(palette)]
        pct = (weight or 0) * 100
        items += (
            f'<div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.3rem;">'
            f'<span style="width:9px; height:9px; border-radius:50%; background:{color}; '
            f'box-shadow:0 0 4px {color}; flex-shrink:0;"></span>'
            f'<span style="font-size:0.76rem; color:{C["text_muted"]};">{label}</span>'
            f'<span style="font-size:0.76rem; color:{C["text_muted"]}; margin-left:auto;">{pct:.0f}%</span>'
            f'</div>'
        )
    return f'<div style="margin-top:0.6rem;">{items}</div>'


def allocation_bars_html(allocation: dict) -> str:
    """Horizontal progress-bar breakdown for a {label: weight_fraction}
    dict - same progress-track/progress-fill pattern as the Market Pulse
    sentiment bars on the Dashboard. Returns a plain string (doesn't call
    st.markdown itself), same contract as svg_donut()."""
    if not allocation:
        return f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No data</span>'
    sorted_items = sorted(allocation.items(), key=lambda kv: kv[1], reverse=True)
    palette = generate_gold_shades(len(sorted_items))
    bars_html = ""
    for i, (label, weight) in enumerate(sorted_items):
        pct = (weight or 0) * 100
        color = palette[i % len(palette)]
        bars_html += (
            f'<div style="margin-bottom:0.5rem;">'
            f'<div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{C["text_muted"]};">'
            f'<span>{label}</span><span>{pct:.0f}%</span></div>'
            f'<div class="progress-track"><div class="progress-fill" style="width:{pct}%; background:{color}; '
            f'box-shadow:0 0 6px {color};"></div></div>'
            f'</div>'
        )
    return bars_html