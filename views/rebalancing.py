import streamlit as st

from views.common import C


def _optimizer_result_html(result: dict, sharpe_label: bool = False) -> str:
    if not result:
        return f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">Not available — run analysis with price history enabled.</span>'

    weights = result.get("weights") or {}
    exp_return = result.get("expected_annual_return")
    exp_vol = result.get("expected_annual_volatility")
    exp_sharpe = result.get("expected_sharpe_ratio")

    stats_html = '<div style="display:flex; gap:1.5rem; margin-bottom:0.7rem; flex-wrap:wrap;">'
    return_display = f"{exp_return:.2%}" if isinstance(exp_return, float) else "N/A"
    vol_display = f"{exp_vol:.2%}" if isinstance(exp_vol, float) else "N/A"
    sharpe_display = f"{exp_sharpe:.2f}" if isinstance(exp_sharpe, float) else "N/A"

    stats_html += (
        f'<div><div style="font-size:0.68rem; color:{C["text_muted"]};">Expected Return</div>'
        f'<div style="font-size:1.0rem; font-weight:600; color:{C["gold_light"]};">{return_display}</div></div>'
    )
    stats_html += (
        f'<div><div style="font-size:0.68rem; color:{C["text_muted"]};">Expected Volatility</div>'
        f'<div style="font-size:1.0rem; font-weight:600; color:{C["gold_light"]};">{vol_display}</div></div>'
    )
    if sharpe_label:
        stats_html += (
            f'<div><div style="font-size:0.68rem; color:{C["text_muted"]};">Expected Sharpe</div>'
            f'<div style="font-size:1.0rem; font-weight:600; color:{C["gold_light"]};">{sharpe_display}</div></div>'
        )
    stats_html += '</div>'

    if weights:
        bars_html = ""
        for ticker, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            pct = (w or 0) * 100
            bars_html += (
                f'<div style="margin-bottom:0.45rem;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{C["text_muted"]};">'
                f'<span>{ticker}</span><span>{pct:.1f}%</span></div>'
                f'<div class="progress-track"><div class="progress-fill" style="width:{max(0,pct)}%; background:{C["gold"]};"></div></div>'
                f'</div>'
            )
    else:
        bars_html = f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">No weight data</span>'

    return stats_html + bars_html


def _target_adjustments_html(recommendations: list) -> str:
    """Target-allocation language only - 'increase/decrease toward target',
    never 'buy/sell' - per README's no-direct-recommendations stance."""
    if not recommendations:
        return (
            f'<span style="color:{C["text_muted"]}; font-size:0.82rem;">'
            f'No target allocations set. Set target weights in Settings (or pass --targets to the CLI) to see gaps here.</span>'
        )
    rows_html = ""
    for rec in recommendations:
        direction = "Increase toward target" if rec.get("action") == "BUY" else "Decrease toward target"
        color = C["green"] if rec.get("action") == "BUY" else C["red_light"]
        current = (rec.get("current_weight") or 0) * 100
        target = (rec.get("target_weight") or 0) * 100
        rows_html += (
            f'<tr><td><b>{rec.get("ticker")}</b></td>'
            f'<td>{current:.1f}%</td><td>{target:.1f}%</td>'
            f'<td style="color:{color};">{direction}</td></tr>'
        )
    return (
        f'<table class="premium-table"><thead><tr>'
        f'<th>Ticker</th><th>Current</th><th>Target</th><th>Direction</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def _cost_estimate_html(cost_estimate: dict) -> str:
    if not cost_estimate:
        return ""
    total = cost_estimate.get("total_estimated_cost")
    note = cost_estimate.get("note") or ""
    total_display = f"₹{total:,.2f}" if isinstance(total, (int, float)) else "N/A"
    return (
        f'<div style="margin-top:0.8rem; padding-top:0.7rem; border-top:1px solid rgba(255,255,255,0.08);">'
        f'<div style="font-size:0.68rem; color:{C["text_muted"]};">Estimated Rebalancing Cost</div>'
        f'<div style="font-size:1.2rem; font-weight:700; color:{C["gold_light"]};">{total_display}</div>'
        f'<p style="font-size:0.75rem; color:{C["text_muted"]}; margin-top:0.3rem;">{note}</p>'
        f'</div>'
    )


def page_rebalancing():
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
    min_var = ph.get("min_variance_portfolio") or {}
    max_sharpe = ph.get("max_sharpe_portfolio") or {}
    recommendations = ph.get("rebalancing_recommendations") or []
    cost_estimate = ph.get("rebalancing_cost_estimate") or {}

    st.markdown('<div class="section-kicker">Rebalancing &amp; Target Allocation</div>', unsafe_allow_html=True)

    # ---- disclaimer banner ----
    st.markdown(
        f'<div class="glass" style="padding:0.7rem 1.2rem; margin-bottom:0.9rem; border-color: rgba(201,165,103,0.3);">'
        f'<span style="font-size:0.8rem; color:{C["gold_light"]};">'
        f'ℹ This page shows portfolio optimization targets and allocation gaps for informational purposes only. '
        f'It is not investment advice and does not recommend specific trades.</span></div>',
        unsafe_allow_html=True,
    )

    # ---- ROW 1: optimizer results ----
    row1 = st.columns([1, 1])
    with row1[0]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Minimum Variance Target</div>'
            f'{_optimizer_result_html(min_var, sharpe_label=False)}</div>',
            unsafe_allow_html=True,
        )
    with row1[1]:
        st.markdown(
            f'<div class="glass" style="padding:1.1rem 1.3rem; height:100%;">'
            f'<div class="section-kicker">Maximum Sharpe Target</div>'
            f'{_optimizer_result_html(max_sharpe, sharpe_label=True)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.9rem;"></div>', unsafe_allow_html=True)

    # ---- ROW 2: target allocation gaps + cost estimate ----
    adjustments_html = _target_adjustments_html(recommendations)
    cost_html = _cost_estimate_html(cost_estimate)
    st.markdown(
        f'<div class="glass" style="padding:1.1rem 1.3rem;">'
        f'<div class="section-kicker">Current vs Target Allocation</div>'
        f'{adjustments_html}{cost_html}</div>',
        unsafe_allow_html=True,
    )