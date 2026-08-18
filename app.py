import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


#page config
st.set_page_config(
    page_title="Nasij",
    page_icon="🧶",
    layout="wide"
)

#load comp vision model

@st.cache_resource
def load_cv_model():
    return YOLO("best_nasij_model.pt")


cv_model = load_cv_model()
#load nlp 
@st.cache_resource
def load_nlp_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

nlp_model = load_nlp_model()
topic_descriptions = {
    "Waste management":
        "garbage collection, trash, waste, garbage bins and dirty streets",

    "Roads / potholes":
        "potholes, damaged roads, road repairs and street maintenance",

    "Street lighting":
        "broken streetlights, dark streets, street lamps and public lighting",

    "Green spaces":
        "trees, parks, gardens, greenery and public green spaces",

    "Water supply":
        "water shortages, water cuts, water distribution and reliable water supply"
}

topic_names = list(topic_descriptions.keys())

topic_embeddings = nlp_model.encode(
    list(topic_descriptions.values()),
    normalize_embeddings=True
)

#header

st.title("🧶 Nasij")
st.subheader("Weaving communities together through AI")

st.write(
    "Nasij helps citizens report infrastructure issues "
    "and share suggestions with municipalities."
)
#initialize 

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "reports" not in st.session_state:
    st.session_state.reports = []

#tabs

tab1, tab2, tab3 = st.tabs([
    "📷 Report an Issue",
    "💬 Submit a Suggestion",
    "🏛️ Municipality Dashboard"
])


#tab1 comp vision

with tab1:

    st.header("Report an Infrastructure Issue")

    st.write(
        "Upload a photo of a civic issue and Nasij will "
        "analyze it automatically."
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.subheader("Uploaded image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🔍 Analyze Issue"):

            with st.spinner("Nasij is analyzing the image..."):

                results = cv_model.predict(
                    source=np.array(image),
                    conf=0.25
                )

                result = results[0]

            if len(result.boxes) == 0:

                st.warning(
                    "Nasij could not confidently detect "
                    "a supported infrastructure issue."
                )

            else:

                annotated_image = result.plot()

                # YOLO returns BGR, Streamlit expects RGB
                annotated_image = annotated_image[:, :, ::-1]

                st.subheader("AI Detection")

                st.image(
                    annotated_image,
                    use_container_width=True
                )

                st.subheader("Detection Results")

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    issue_name = cv_model.names[class_id]

                    st.success(
                        f"Detected: {issue_name.replace('_', ' ').title()} "
                        f"— Confidence: {confidence:.1%}"
                    )
                    st.session_state.reports.append({
                        "issue": issue_name.replace("_", " ").title(),
                        "confidence": confidence
                    })


#tab2 nlp

with tab2:

    st.header("Submit a Suggestion")

    st.write(
        "Share an idea or concern with your municipality."
    )

    suggestion = st.text_area(
        "Your suggestion",
        placeholder="Example: The potholes near our neighborhood need urgent repair."
    )

    if st.button("Submit Suggestion", type="primary"):

        if suggestion.strip():

            suggestion_embedding = nlp_model.encode(
                [suggestion],
                normalize_embeddings=True
            )

            similarities = cosine_similarity(
                suggestion_embedding,
                topic_embeddings
            )[0]

            best_index = similarities.argmax()

            detected_topic = topic_names[best_index]
            confidence = similarities[best_index]

            st.session_state.suggestions.append({
                "suggestion": suggestion,
                "topic": detected_topic,
                "confidence": float(confidence)
            })

            st.success("Suggestion submitted successfully! ✅")

            st.write(f"**Detected topic:** {detected_topic}")
            st.write(f"**Semantic similarity:** {confidence:.1%}")

        else:
            st.warning("Please enter a suggestion before submitting.")


#tab3 dashboard

with tab3:

    st.header("🏛️ Municipality Dashboard")

    st.write(
        "Monitor reported infrastructure issues "
        "and understand citizens' most common concerns."
    )

  #overview

    total_suggestions = len(st.session_state.suggestions)
    total_reports = len(st.session_state.reports)

    if total_suggestions > 0:

        suggestion_topics = [
            item["topic"]
            for item in st.session_state.suggestions
        ]

        topic_counts = pd.Series(
            suggestion_topics
        ).value_counts()

        top_priority = topic_counts.index[0]

    else:
        topic_counts = pd.Series(dtype=int)
        top_priority = "No data yet"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📷 Infrastructure Reports",
            total_reports
        )

    with col2:
        st.metric(
            "💬 Citizen Suggestions",
            total_suggestions
        )

    with col3:
        st.metric(
            "🔥 Top Citizen Concern",
            top_priority
        )

    st.divider()

    #feedback analysis

    st.subheader("💬 Citizen Feedback Analysis")

    if total_suggestions == 0:

        st.info(
            "No suggestions have been submitted yet."
        )

    else:

        st.write("### Most Requested Topics")

        topic_dataframe = topic_counts.reset_index()

        topic_dataframe.columns = [
            "Topic",
            "Number of Suggestions"
        ]

        st.dataframe(
            topic_dataframe,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            topic_dataframe.set_index("Topic")
        )

        st.success(
            f"🔥 Current top priority: **{top_priority}**"
        )

        st.write("### Submitted Suggestions")

        for item in reversed(st.session_state.suggestions):

            with st.container(border=True):

                st.write(
                    f"**Topic:** {item['topic']}"
                )

                st.write(
                    f"**Suggestion:** {item['suggestion']}"
                )

                st.caption(
                    f"Semantic similarity: "
                    f"{item['confidence']:.1%}"
                )

    st.divider()

    #infrastructure reports

    st.subheader("📷 Infrastructure Reports")

    if total_reports == 0:

        st.info(
            "No infrastructure reports have been submitted yet."
        )

    else:

        for report in reversed(st.session_state.reports):

            with st.container(border=True):

                st.write(
                    f"**Issue:** {report['issue']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{report['confidence']:.1%}"
                )

                st.write(
                    "**Status:** 🟡 Pending"
                )
