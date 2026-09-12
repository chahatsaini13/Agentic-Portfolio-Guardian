import streamlit as st

from views.common import C

def page_placeholder(name: str):
    st.markdown(
        f'<div class="glass" style="padding:3rem 2rem; text-align:center;">'
        f'<div class="section-kicker">{name}</div>'
        f'<h2 style="color:{C["text_muted"]}; font-weight:500;">Not built yet</h2>'
        f'<p style="color:{C["text_muted"]}; font-size:0.85rem;">This view is part of the design system but isn\'t wired up yet. Ask to build the {name} page next.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )