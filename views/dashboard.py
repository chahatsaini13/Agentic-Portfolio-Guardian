from collections import Counter

import streamlit as st

from views.common import C, SENTIMENT_COLOR, badge, svg_donut


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