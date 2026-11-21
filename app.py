"""ASL Vision -- Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="ASL Vision", page_icon="🤟", layout="wide", initial_sidebar_state="expanded")

from pages_impl import about, comparison, methodology, overview, recognition  # noqa: E402
from ui.theme import inject  # noqa: E402

inject()

pages = {
    "overview": st.Page(overview.render, title="Project Overview", icon=":material/home:", url_path="overview",
                        default=True),
    "recognition": st.Page(recognition.render, title="Live Recognition", icon=":material/videocam:",
                           url_path="recognition"),
    "comparison": st.Page(comparison.render, title="Model Comparison", icon=":material/insights:",
                          url_path="comparison"),
    "methodology": st.Page(methodology.render, title="Methodology", icon=":material/schema:",
                           url_path="methodology"),
    "about": st.Page(about.render, title="About", icon=":material/groups:", url_path="about"),
}
st.session_state["_pages"] = pages

with st.sidebar:
    st.markdown("<div style='font-size:1.45rem;font-weight:800;margin:.2rem 0 0'>🤟 ASL Vision</div>"
                "<div style='opacity:.75;font-size:.85rem;margin-bottom:.6rem'>Isolated ASL sign recognition"
                "</div>", unsafe_allow_html=True)

nav = st.navigation(list(pages.values()))
with st.sidebar:
    st.markdown("<div style='opacity:.6;font-size:.75rem;margin-top:1.5rem'>Capstone project · East West University"
                "<br>CPU inference · ASL Citizen</div>", unsafe_allow_html=True)
nav.run()
