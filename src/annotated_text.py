import streamlit as st
from typing import Union

def annotated_text(*args):
    """Ermöglicht das farbliche Markieren von Textteilen in Streamlit."""
    out = []
    for arg in args:
        if isinstance(arg, str):
            out.append(arg)
        elif isinstance(arg, tuple):
            text, label, color = arg
            out.append(f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; margin: 0 2px;">{text} <small style="opacity: 0.7;">{label}</small></span>')
    st.markdown(" ".join(out), unsafe_allow_html=True)
