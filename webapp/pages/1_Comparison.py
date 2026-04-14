import streamlit as st
import pandas as pd

st.set_page_config(page_title="Comparison Dashboard", layout="wide")

import os

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.markdown('<h1 class="main-header">Comparison History</h1>', unsafe_allow_html=True)
st.write("This page compares angle encoding and amplitude encoding results. It provides a historical overview of all your custom molecule predictions so you can compare attributes side-by-side.")
st.markdown("---")

if st.session_state.get('prediction_history'):
    df_history = pd.DataFrame(st.session_state.prediction_history)
    
    # Optional styling for table
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    if st.button("🧹 Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("No predictions have been made yet! Head over to the main app, start an experiment, and run some custom molecule predictions to see them compared here.")
