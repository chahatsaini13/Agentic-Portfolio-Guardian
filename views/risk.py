import streamlit as st

from views.common import C, badge


def _metric_tile(label: str, val, as_pct: bool = True) -> str:
    if val is None:
        display = "N/A"
    elif as_pct:
        display = f"{val:.2%}"
    else:
        display = f"{val:.4f}" if isinstance(val, float) else f"{val}"
    return (
        f'<div><div style="font-size:0.68rem; color:{C["text_muted"]};">{label}</div>'
        f'<div style="font-size:1.05rem; font-weight:600; color:{C["gold_light"]};">{display}</div></div>'
    )


def _correlation_table_html(correlation_matrix: dict) -> str:
    if not correlation_matrix:
        return f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No correlation data.</span>'
    tickers = list(correlation_matrix.keys())
    header = "<th></th>" + "".join(f"<th>{t}</th>" for t in tickers)
    rows_html = ""
    for row_ticker in tickers:
        cells = "".join(
            f"<td>{correlation_matrix.get(col_ticker, {}).get(row_ticker, ''):.2f}</td>"
            if isinstance(correlation_matrix.get(col_ticker, {}).get(row_ticker), (int, float)) else "<td>—</td>"
            for col_ticker in tickers
        )
        rows_html += f"<tr><td><b>{row_ticker}</b></td>{cells}</tr>"
    return f'<table class="premium-table"><thead><tr>{header}</tr></thead><tbody>{rows_html}</tbody></table>'


def _flagged_items_html(exposure_alerts: list, liquidity_flags: list, position_size_flags: list) -> str:
    items = []
    for a in exposure_alerts:
        label = a.get("ticker") or a.get("sector") or a.get("theme")
        weight = (a.get("weight") or 0) * 100
        limit = (a.get("limit") or 0) * 100
        items.append(f'{badge(a.get("type"), "weakening")} {label} — {weight:.1f}% (limit {limit:.0f}%)')
    for f in liquidity_flags:
        items.append(f'{badge("low liquidity", "broken")} {f.get("ticker")} — avg volume {f.get("avg_volume")}')
    for f in position_size_flags:
        weight = (f.get("weight") or 0) * 100
        issue_label = (f.get("issue") or "").replace("_", " ")
        items.append(f'{badge(issue_label, "weakening")} {f.get("ticker")} — {weight:.1f}%')

    if not items:
        return f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No flagged items.</span>'
    return "".join(f'<div style="margin-bottom:0.5rem; font-size:0.85rem;">{item}</div>' for item in items)


def page_risk():
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
    risk = ph.get("risk_metrics") or {}
    risk_contribution = risk.get("risk_contribution_by_holding") or {}
    correlation_matrix = ph.get("correlation_matrix") or {}
    exposure_alerts = ph.get("exposure_alerts") or []
    liquidity_flags = ph.get("liquidity_flags") or []
    position_size_flags = ph.get("position_size_flags") or []

    st.markdown('<div class="section-kicker">Risk &amp; Exposure</div>', unsafe_allow_html=True)

    # ---- ROW 1: key risk metric tiles ----
    if risk:
        tiles_html = '<div style="display:flex; gap:1.8rem; flex-wrap:wrap; margin-top:0.4rem;">'
        tiles_html += _metric_tile("Volatility (ann.)", risk.get("volatility_annualized"))
        tiles_html += _metric_tile("Beta vs Benchmark", risk.get("beta_vs_benchmark"), as_pct=False)
        tiles_html += _metric_tile("Max Drawdown", risk.get("max_drawdown"))
        tiles_html += _metric_tile("VaR (95%)", risk.get("value_at_risk_95"))
        tiles_html += _metric_tile("CVaR (95%)", risk.get("conditional_var_95"))
        tiles_html += _metric_tile("Downside Deviation", risk.get("downside_deviation"))
        tiles_html += '</div>'
        empty_note = ''
    else:
        tiles_html = ''
        empty_note = (
            f'<p style="font-size:0.8rem; color:{C["text_muted"]};">'
            f'No risk metrics available — run analysis with price history enabled (Settings page).</p>'
        )

    st.markdown(
        f'<div class="glass" style="padding:1.2rem 1.4rem;">'
        f'<div class="section-kicker">Portfolio Risk Metrics</div>{tiles_html}{empty_note}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    # ---- ROW 2: risk contribution + flagged items ----
    row2 = st.columns([1, 1])
    with row2[0]:
        if risk_contribution:
            bars_html = ""
            for ticker, contrib in sorted(risk_contribution.items(), key=lambda kv: kv[1], reverse=True):
                pct = (contrib or 0) * 100
                bars_html += (
                    f'<div style="margin-bottom:0.5rem;">'
                    f'<div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{C["text_muted"]};">'
                    f'<span>{ticker}</span><span>{pct:.1f}%</span></div>'
                    f'<div class="progress-track"><div class="progress-fill" style="width:{max(0,pct)}%; background:{C["gold"]};"></div></div>'
                    f'</div>'
                )
        else:
            bars_html = f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No data</span>'
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Risk Contribution by Holding</div>{bars_html}</div>',
            unsafe_allow_html=True,
        )

    with row2[1]:
        flagged_html = _flagged_items_html(exposure_alerts, liquidity_flags, position_size_flags)
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Flagged Items</div>{flagged_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    # ---- CORRELATION MATRIX ----
    corr_html = _correlation_table_html(correlation_matrix)
    st.markdown(
        f'<div class="glass" style="padding:1.1rem 1.3rem;">'
        f'<div class="section-kicker">Correlation Matrix</div>{corr_html}</div>',
        unsafe_allow_html=True,
    )