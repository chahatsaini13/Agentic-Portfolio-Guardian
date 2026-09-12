import os
import tempfile

import streamlit as st

from views.common import (
    DEFAULT_PORTFOLIO_PATH, THESES_PATH,
    load_theses_file, save_theses_file, load_portfolio, run_analysis,
)


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