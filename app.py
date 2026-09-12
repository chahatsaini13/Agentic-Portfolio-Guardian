"""
Agentic Portfolio Guardian - Streamlit app (design-system rebuild)

Visual language: dark wine/burgundy atmosphere, translucent smoked-glass
panels, muted champagne gold accents, restrained crimson for alerts, clean
sans-serif (Inter) throughout. Dashboard page is single-screen/no-scroll;
Thesis Monitor is the scrollable detail page.

Page implementations live in views/ - one file per page - imported below.
Shared design tokens/helpers live in views/common.py.

Usage:
    streamlit run app.py
"""

import streamlit as st

from views.common import C
from views.dashboard import page_dashboard
from views.thesis import page_thesis
from views.settings import page_settings
from views.placeholder import page_placeholder
from views.portfolio import page_portfolio
from views.risk import page_risk
from views.rebalancing import page_rebalancing

st.set_page_config(page_title="Portfolio Intelligence", page_icon="🍷", layout="wide")

NAV_ITEMS = ["Dashboard", "Portfolio", "Risk", "Thesis", "Markets", "Early Warning", "Rebalancing", "Reports", "Settings"]
NAV_ITEMS = ["Dashboard", "Portfolio", "Risk", "Thesis", "Markets", "Early Warning", "Rebalancing", "Reports", "Settings"]
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

    section[data-testid="stSidebar"] {{
        transform: none !important;
        visibility: visible !important;
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        transform: none !important;
    }}
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

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

    div[data-testid="stButton"] button[kind="primary"] {{
        background: linear-gradient(135deg, {C['red_light']}, {C['red_deep']}) !important;
        border: 1px solid rgba(192,90,98,0.5) !important;
        border-radius: 999px !important; font-weight: 600 !important; font-size: 0.82rem !important;
        padding: 0.4rem 1.1rem !important;
    }}

    .section-kicker {{ font-size: 0.68rem; letter-spacing: 0.09em; text-transform: uppercase; color: {C['text_muted']}; margin-bottom: 0.5rem; }}

    .badge {{ display: inline-block; font-size: 0.76rem; font-weight: 600; padding: 0.24rem 0.7rem; border-radius: 999px; }}
    .badge.holds     {{ background: rgba(143,163,122,0.14); color: {C['green']}; border: 1px solid rgba(143,163,122,0.35); }}
    .badge.weakening {{ background: rgba(201,165,103,0.14); color: {C['gold']}; border: 1px solid rgba(201,165,103,0.35); }}
    .badge.broken    {{ background: rgba(168,60,70,0.16); color: {C['red_light']}; border: 1px solid rgba(168,60,70,0.4); }}
    .badge.neutral   {{ background: rgba(255,255,255,0.05); color: {C['text_muted']}; border: 1px solid rgba(255,255,255,0.1); }}

    .progress-track {{ background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden; }}
    .progress-fill {{ height: 100%; border-radius: 999px; }}

    table.premium-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    table.premium-table th {{ text-align: left; color: {C['text_muted']}; font-weight: 500; padding: 0.45rem 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    table.premium-table td {{ padding: 0.55rem 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    table.premium-table tr:hover td {{ background: rgba(201,165,103,0.04); }}
    table.premium-table tr:last-child td {{ border-bottom: none; }}

    [data-testid="stExpander"] {{
        background: {C['glass_bg']}; border: 1px solid {C['glass_border']}; border-radius: 14px;
        backdrop-filter: blur(20px);
    }}

    .scroll-box {{ overflow-y: auto; }}
    .scroll-box::-webkit-scrollbar {{ width: 6px; }}
    .scroll-box::-webkit-scrollbar-thumb {{ background: rgba(201,165,103,0.28); border-radius: 999px; }}
    </style>
    """, unsafe_allow_html=True)


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


def render_header():
    from views.common import run_analysis
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
elif page == "Portfolio":
    page_portfolio()
elif page == "Risk":
    page_risk()
elif page == "Rebalancing":
    page_rebalancing()
else:
    page_placeholder(page)