import streamlit as st

from views.common import C, badge, severity_kind

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _alert_card_html(alert: dict) -> str:
    """One alert, one combined glass box - open/content/close in a single
    string, same pattern as views/thesis.py's alert boxes."""
    sev = alert.get("severity")
    flag_label = f'[{alert.get("flag_type") or "flag"}] {sev or "N/A"}'
    agreement = alert.get("signal_agreement")
    agreement_line = (
        f'<div style="margin-top:0.3rem; font-size:0.72rem; color:{C["text_muted"]};">Signal agreement: {agreement}</div>'
        if agreement else ""
    )
    return (
        f'<div class="glass" style="padding:0.9rem 1.1rem; margin-bottom:0.7rem; border-color:rgba(168,60,70,0.4);">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="font-weight:600; font-size:0.9rem;">{alert.get("ticker") or "N/A"}</span>'
        f'{badge(flag_label, severity_kind(sev))}'
        f'</div>'
        f'<div style="margin-top:0.5rem; font-size:0.85rem;">{alert.get("headline") or "(no headline)"}</div>'
        f'<div style="margin-top:0.35rem; font-size:0.8rem; color:{C["text_muted"]};">{alert.get("reasoning") or ""}</div>'
        f'{agreement_line}'
        f'</div>'
    )


def page_early_warning():
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

    per_holding = (final_state.get("final_output") or {}).get("per_holding") or []

    # Flatten every holding's redflag_alerts into one combined feed,
    # tagging each alert with its ticker so it's traceable back.
    all_alerts = []
    for h in per_holding:
        for alert in (h.get("redflag_alerts") or []):
            all_alerts.append({**alert, "ticker": alert.get("ticker") or h.get("ticker")})

    all_alerts.sort(key=lambda a: _SEVERITY_ORDER.get(a.get("severity"), 3))

    n_high = sum(1 for a in all_alerts if a.get("severity") == "HIGH")
    n_medium = sum(1 for a in all_alerts if a.get("severity") == "MEDIUM")
    n_low = sum(1 for a in all_alerts if a.get("severity") == "LOW")

    st.markdown('<div class="section-kicker">Early Warning — Portfolio-Wide Alerts Feed</div>', unsafe_allow_html=True)

    summary_chips = (
        f'{badge(f"{n_high} High", "broken")} &nbsp; '
        f'{badge(f"{n_medium} Medium", "weakening")} &nbsp; '
        f'{badge(f"{n_low} Low", "holds")}'
    )
    st.markdown(
        f'<div class="glass" style="padding:0.9rem 1.2rem; margin-bottom:1rem;">'
        f'<div style="margin-bottom:0.3rem;">{summary_chips}</div>'
        f'<span style="font-size:0.78rem; color:{C["text_muted"]};">{len(all_alerts)} total alert(s) across {len(per_holding)} holding(s), sorted by severity.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not all_alerts:
        st.markdown(
            f'<div class="glass" style="padding:1.5rem; text-align:center;">'
            f'<span style="color:{C["text_muted"]}; font-size:0.85rem;">No red-flag alerts on any holding.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    for alert in all_alerts:
        st.markdown(_alert_card_html(alert), unsafe_allow_html=True)
