import streamlit as st

from views.common import C, SENTIMENT_COLOR, badge


def _sentiment_kind(sentiment: str) -> str:
    """Maps a sentiment string onto the same badge color classes
    (holds/weakening/broken/neutral) status_kind()/severity_kind() use,
    so sentiment badges look consistent with the rest of the app."""
    return {"positive": "holds", "negative": "broken", "neutral": "neutral"}.get(sentiment, "neutral")


def _news_item_html(item: dict) -> str:
    """One article, one combined glass box - open/content/close in a
    single string, same pattern as views/thesis.py's alert boxes."""
    sentiment = item.get("sentiment") or "neutral"
    source = item.get("source") or "Unknown source"
    published = item.get("published_at") or ""
    return (
        f'<div class="glass" style="padding:0.7rem 0.9rem; margin-top:0.5rem; border-color:rgba(180,150,100,0.15);">'
        f'{badge(sentiment.capitalize(), _sentiment_kind(sentiment))}'
        f'<div style="margin-top:0.4rem; font-size:0.85rem;">{item.get("title") or "(no title)"}</div>'
        f'<div style="margin-top:0.3rem; font-size:0.75rem; color:{C["text_muted"]};">{source} · {published}</div>'
        f'</div>'
    )


def page_markets():
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
    st.markdown('<div class="section-kicker">Market Intelligence — Per-Holding News &amp; Sentiment</div>', unsafe_allow_html=True)

    if not per_holding:
        st.markdown(
            f'<div class="glass" style="padding:1.5rem; text-align:center;">'
            f'<span style="color:{C["text_muted"]}; font-size:0.85rem;">No holdings in the current analysis.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    for h in per_holding:
        ticker = h.get("ticker")
        sentiment = h.get("market_sentiment") or "neutral"
        summary = h.get("market_summary") or "_No summary available._"
        news_items = h.get("news_items") or []

        with st.expander(f"{ticker} — {sentiment.capitalize()} ({len(news_items)} article(s))"):
            st.markdown(badge(sentiment.capitalize(), _sentiment_kind(sentiment)), unsafe_allow_html=True)
            st.markdown(f"**Summary:** {summary}")
            st.markdown("---")
            if news_items:
                st.markdown(f"**Recent articles ({len(news_items)}):**")
                for item in news_items:
                    st.markdown(_news_item_html(item), unsafe_allow_html=True)
            else:
                st.markdown("**Recent articles:** none found for this holding.")
