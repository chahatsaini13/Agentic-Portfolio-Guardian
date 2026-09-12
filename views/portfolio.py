import streamlit as st

from views.common import C, DEFAULT_PORTFOLIO_PATH, allocation_bars_html, load_portfolio, svg_donut


def page_portfolio():
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

    ph = (final_state.get("final_output") or {}).get("portfolio_health") or {}
    sector_weights = ph.get("sector_weights") or {}
    asset_class_allocation = ph.get("asset_class_allocation") or {}
    industry_allocation = ph.get("industry_allocation") or {}
    market_cap_allocation = ph.get("market_cap_allocation") or {}
    thematic_overlap = ph.get("thematic_overlap") or {}

    st.markdown('<div class="section-kicker">Portfolio Composition</div>', unsafe_allow_html=True)

    row1 = st.columns([1, 1, 1])
    with row1[0]:
        if sector_weights:
            body = (
                f'<div style="display:flex; justify-content:center; margin:0.2rem 0 0.4rem;">{svg_donut(sector_weights)}</div>'
                f'<div style="font-size:0.78rem; color:{C["text_muted"]}; text-align:center;">By sector</div>'
            )
        else:
            body = f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No sector data</span>'
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Sector Allocation</div>{body}</div>',
            unsafe_allow_html=True,
        )

    with row1[1]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Asset Class</div>{allocation_bars_html(asset_class_allocation)}</div>',
            unsafe_allow_html=True,
        )

    with row1[2]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Market Cap</div>{allocation_bars_html(market_cap_allocation)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    row2 = st.columns([1, 1])
    with row2[0]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Industry</div>{allocation_bars_html(industry_allocation)}</div>',
            unsafe_allow_html=True,
        )
    with row2[1]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Thematic Overlap</div>{allocation_bars_html(thematic_overlap)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    holdings = st.session_state.get("holdings_cache") or []
    if not holdings:
        try:
            holdings = load_portfolio(st.session_state.get("portfolio_path", DEFAULT_PORTFOLIO_PATH))
        except Exception:
            holdings = []

    if holdings:
        rows_html = "".join(
            f'<tr><td>{h["ticker"]}</td><td>{h["sector"]}</td><td>{h["weight"]*100:.1f}%</td></tr>'
            for h in holdings
        )
        table_html = (
            f'<table class="premium-table"><thead><tr>'
            f'<th>Ticker</th><th>Sector</th><th>Weight</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )
    else:
        table_html = f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No holdings loaded.</span>'

    st.markdown(
        f'<div class="glass" style="padding:1.1rem 1.3rem;">'
        f'<div class="section-kicker">Holdings</div>{table_html}</div>',
        unsafe_allow_html=True,
    )