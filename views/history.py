import streamlit as st
from views.common import C, load_theses_file


def load_history(ticker):
    return [
        {"timestamp": "2026-11-18 10:00", "verdict": "HOLDS", "key_changes": "____"},
        {"timestamp": "2026-11-19 10:00", "verdict": "____", "key_changes": "____"},
    ]


def page_history():
    st.markdown('<div class="section-kicker">____</div>', unsafe_allow_html=True)
    tickers = list(load_theses_file().keys())
    ticker = st.selectbox("____", tickers)
    rows = load_history(ticker)
    if not rows:
        st.info("____")
        return
    st.dataframe(rows, use_container_width=True)