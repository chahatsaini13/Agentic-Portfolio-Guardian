import streamlit as st

from views.common import badge, severity_kind, status_kind


def page_thesis():
    final_state = st.session_state.get("final_state")
    if not final_state:
        from views.common import C
        st.markdown(
            f'<div class="glass" style="padding:2.5rem; text-align:center;">'
            f'<p style="color:{C["text_muted"]}; margin:0;">No analysis yet — run it from Settings first.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    from views.common import C

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