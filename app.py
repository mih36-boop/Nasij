import streamlit as st

st.set_page_config(
    page_title="Nasij",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Nasij")
st.subheader("Weaving communities together through AI")

st.write(
    "Nasij helps citizens report infrastructure issues "
    "and share suggestions with municipalities."
)

tab1, tab2, tab3 = st.tabs([
    "📷 Report an Issue",
    "💬 Submit a Suggestion",
    "🏛️ Municipality Dashboard"
])

with tab1:
    st.header("Report an Infrastructure Issue")
    st.write("Upload a photo of a civic issue.")

with tab2:
    st.header("Submit a Suggestion")
    st.write("Share an idea or concern with your municipality.")

with tab3:
    st.header("Municipality Dashboard")
    st.write("View citizen reports and the most common concerns.")
