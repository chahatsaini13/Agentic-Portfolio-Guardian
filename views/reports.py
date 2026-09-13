import os
import tempfile

import streamlit as st

from views.common import C
from src.agents.portfolio_health_agent import export_report


def page_reports():
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

    final_output = final_state.get("final_output") or {}
    last_run = st.session_state.get("last_run_ts") or "unknown time"

    st.markdown('<div class="section-kicker">Reports — Export Full Analysis</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="glass" style="padding:1.1rem 1.3rem; margin-bottom:1rem;">'
        f'<p style="font-size:0.85rem; color:{C["text"]}; margin:0;">'
        f'Download the merged output from the last run ({last_run}) - portfolio health, '
        f'per-holding thesis status, market sentiment, and red-flag alerts, in one file.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    tmp_dir = tempfile.gettempdir()

    with col1:
        json_path = export_report(final_output, os.path.join(tmp_dir, "portfolio_guardian_report.json"), "json")
        with open(json_path, "rb") as f:
            st.download_button(
                "Download JSON", data=f.read(),
                file_name="portfolio_guardian_report.json", mime="application/json",
                use_container_width=True,
            )

    with col2:
        csv_path = export_report(final_output, os.path.join(tmp_dir, "portfolio_guardian_report.csv"), "csv")
        with open(csv_path, "rb") as f:
            st.download_button(
                "Download CSV", data=f.read(),
                file_name="portfolio_guardian_report.csv", mime="text/csv",
                use_container_width=True,
            )

    st.markdown(
        f'<p style="font-size:0.72rem; color:{C["text_muted"]}; margin-top:0.8rem;">'
        f'CSV export flattens nested sections (portfolio health, per-holding) into JSON-stringified '
        f'cells - useful for archiving, not for spreadsheet analysis. Use JSON for anything downstream.</p>',
        unsafe_allow_html=True,
    )
