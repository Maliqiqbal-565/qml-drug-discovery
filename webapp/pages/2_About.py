import os
import streamlit as st

st.set_page_config(page_title="About QML Drug Discovery", layout="wide")


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


load_css()

st.markdown('<h1 class="main-header">About Our Platform</h1>',
            unsafe_allow_html=True)

st.markdown("### Quantum Machine Learning for Drug Discovery")
st.write("This platform is built using cutting-edge Quantum Machine Learning (QML) libraries including **Qiskit**, **PennyLane**, and **Streamlit** to revolutionize how we approach molecular property prediction.")

st.markdown("---")

st.markdown("### 🌟 Why is this Website Important?")
st.write("""
Traditional machine learning and classical computers inherently struggle with the exponentially complex nature of quantum chemistry. Accurately predicting the properties of a molecule (such as its aqueous solubility) is a critical hurdle for screening potential life-saving drugs *before* they ever enter a physical laboratory. 

By aggressively leveraging **Quantum Encoding techniques** (like Angle and Amplitude mapping), this application pushes beyond classical computing boundaries. It serves as a proof-of-concept for analyzing complex chemical spaces hyper-efficiently, paving a pathway to potentially save pharmaceutical researchers years of trial-and-error and millions of dollars in R&D costs.
""")

st.markdown("### 🚀 Key Platform Advantages")
st.markdown("""
- **Interactive Predictive Analytics:** Test any arbitrary chemical dynamically in seconds, integrated directly with the global PubChem API for seamless structure lookups.
- **Side-by-Side Quantum Benchmarking:** Directly pit varied state-of-the-art quantum encodings (Angle vs. Amplitude) against one another to empirically view which models learn faster on molecular data.
- **Accessible & Intuitive UI:** Built intentionally for both data scientists and life scientists, bridging the gap between intimidating quantum physics terminal scripts and a beautiful, friendly dashboard.
- **Future-Ready Foundation:** The math and tensors utilized within this platform are structured to seamlessly execute on next-generation physical quantum hardware, rather than being permanently locked to simulators.
""")
