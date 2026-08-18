import streamlit as st
import pandas as pd
import numpy as np
import re

from PIL import Image
from datetime import datetime

from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Nasij",
    page_icon="🧶",
    layout="wide"
)


@st.cache_resource
def load_cv_model():
    return YOLO("best_nasij_model.pt")


@st.cache_resource
def load_nlp_resources():

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

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

    topic_embeddings = model.encode(
        list(topic_descriptions.values()),
        normalize_embeddings=True
    )

    return model, topic_names, topic_embeddings


def normalize_arabic(text):

    text = text.lower()

    text = re.sub(
        r"[\u064B-\u065F\u0670\u0640]",
        "",
        text
    )

    text = (
        text
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )

    return text


arabic_keywords = {

    "Waste management": [
        "زبال",
        "نفا",
        "قمام",
        "حاوي"
    ],

    "Roads / potholes": [
        "حفر",
        "حفره",
        "طريق",
        "طرقات",
        "زفت",
        "اسفلت"
    ],

    "Street lighting": [
        "انار",
        "ضو",
        "عتم",
        "لمب",
        "عواميد"
    ],

    "Green spaces": [
        "شجر",
        "حديق",
        "خضر",
        "منتزه"
    ],

    "Water supply": [
        "المي",
        "مياه",
        "ماء",
        "مي عم",
        "توزيع المي",
        "انقطاع المي"
    ]
}


def classify_suggestion(
    suggestion,
    nlp_model,
    topic_names,
    topic_embeddings
):

    normalized_text = normalize_arabic(
        suggestion
    )

    keyword_scores = {}

    for topic, keywords in arabic_keywords.items():

        score = 0

        for keyword in keywords:

            normalized_keyword = normalize_arabic(
                keyword
            )

            if normalized_keyword in normalized_text:
                score += 1

        keyword_scores[topic] = score

    highest_keyword_score = max(
        keyword_scores.values()
    )

    if highest_keyword_score > 0:

        best_topics = [
            topic
            for topic, score
            in keyword_scores.items()
            if score == highest_keyword_score
        ]

        if len(best_topics) == 1:

            return (
                best_topics[0],
                None,
                "Lebanese Arabic keyword match"
            )

    suggestion_embedding = nlp_model.encode(
        [suggestion],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        suggestion_embedding,
        topic_embeddings
    )[0]

    best_index = similarities.argmax()

    detected_topic = topic_names[
        best_index
    ]

    match_score = float(
        similarities[best_index]
    )

    return (
        detected_topic,
        match_score,
        "Multilingual semantic similarity"
    )


def find_representative_feedback(
    topic,
    suggestions,
    nlp_model,
    topic_names,
    topic_embeddings
):

    if len(suggestions) == 1:
        return suggestions[0]

    embeddings = nlp_model.encode(
        suggestions,
        normalize_embeddings=True
    )

    topic_index = topic_names.index(
        topic
    )

    prototype = (
        topic_embeddings[topic_index]
        .reshape(1, -1)
    )

    similarities = cosine_similarity(
        embeddings,
        prototype
    ).flatten()

    best_index = similarities.argmax()

    return suggestions[best_index]


if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "reports" not in st.session_state:
    st.session_state.reports = []

if "next_ticket_id" not in st.session_state:
    st.session_state.next_ticket_id = 1


with st.sidebar:

    st.header("Nasij")

    st.write(
        "AI-powered civic reporting platform."
    )

    if st.button(
        "🗑️ Reset Demo Data"
    ):

        st.session_state.suggestions = []
        st.session_state.reports = []
        st.session_state.next_ticket_id = 1

        st.success(
            "Demo data cleared."
        )

        st.rerun()


st.title("🧶 Nasij")

st.subheader(
    "Weaving communities together through AI"
)

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

    st.header(
        "📷 Report an Infrastructure Issue"
    )

    st.write(
        "Upload a photo of a civic issue and "
        "Nasij will analyze it automatically."
    )

    location = st.text_input(
        "Location / Area",
        placeholder="Example: Hamra, Beirut"
    )

    issue_note = st.text_area(
        "Additional details (optional)",
        placeholder=(
            "Example: This pothole has been "
            "causing problems for cars."
        )
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.subheader(
            "Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Issue",
            type="primary"
        ):

            with st.spinner(
                "Nasij is analyzing the image..."
            ):

                cv_model = load_cv_model()

                results = cv_model.predict(
                    source=np.array(image),
                    conf=0.25,
                    verbose=False
                )

                result = results[0]

            if len(result.boxes) == 0:

                st.warning(
                    "Nasij could not confidently "
                    "detect a supported infrastructure issue."
                )

            else:

                annotated_image = result.plot()

                annotated_image = (
                    annotated_image[:, :, ::-1]
                )

                st.subheader(
                    "AI Detection"
                )

                st.image(
                    annotated_image,
                    use_container_width=True
                )

                detections = []

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    issue_name = (
                        cv_model.names[class_id]
                    )

                    readable_name = (
                        issue_name
                        .replace("_", " ")
                        .title()
                    )

                    detections.append({
                        "issue":
                            readable_name,

                        "confidence":
                            confidence
                    })

                    st.success(
                        f"Detected: "
                        f"**{readable_name}** "
                        f"— AI confidence: "
                        f"**{confidence:.1%}**"
                    )

                primary_detection = max(
                    detections,
                    key=lambda x:
                        x["confidence"]
                )

                ticket_id = (
                    f"NAS-"
                    f"{st.session_state.next_ticket_id:03d}"
                )

                st.session_state.next_ticket_id += 1

                report = {

                    "ticket_id":
                        ticket_id,

                    "issue":
                        primary_detection[
                            "issue"
                        ],

                    "confidence":
                        primary_detection[
                            "confidence"
                        ],

                    "location":
                        location.strip()
                        if location.strip()
                        else "Not specified",

                    "note":
                        issue_note.strip()
                        if issue_note.strip()
                        else "No additional details",

                    "status":
                        "Pending",

                    "time":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        )
                }

                st.session_state.reports.append(
                    report
                )

                st.success(
                    f"✅ Maintenance ticket "
                    f"**{ticket_id}** created."
                )

                st.write(
                    f"**Issue:** "
                    f"{report['issue']}"
                )

                st.write(
                    f"**Location:** "
                    f"{report['location']}"
                )

                st.write(
                    "**Status:** 🟡 Pending"
                )


with tab2:

    st.header(
        "💬 Submit a Suggestion"
    )

    st.write(
        "Share an idea or concern with your "
        "municipality. Nasij will automatically "
        "determine the civic topic."
    )

    suggestion = st.text_area(
        "Your suggestion",
        placeholder=(
            "Example: The potholes near our "
            "neighborhood need urgent repair."
        ),
        height=150
    )

    if st.button(
        "Submit Suggestion",
        type="primary"
    ):

        if not suggestion.strip():

            st.warning(
                "Please enter a suggestion "
                "before submitting."
            )

        else:

            with st.spinner(
                "Nasij is analyzing your suggestion..."
            ):

                (
                    nlp_model,
                    topic_names,
                    topic_embeddings
                ) = load_nlp_resources()

                (
                    detected_topic,
                    match_score,
                    detection_method
                ) = classify_suggestion(
                    suggestion,
                    nlp_model,
                    topic_names,
                    topic_embeddings
                )

            new_suggestion = {

                "suggestion":
                    suggestion.strip(),

                "topic":
                    detected_topic,

                "match_score":
                    match_score,

                "method":
                    detection_method,

                "time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
            }

            st.session_state.suggestions.append(
                new_suggestion
            )

            st.success(
                "Suggestion submitted successfully! ✅"
            )

            st.write(
                f"**Detected topic:** "
                f"{detected_topic}"
            )

            if match_score is not None:

                st.write(
                    f"**Topic match score:** "
                    f"{match_score:.1%}"
                )

                st.caption(
                    "Topic identified using "
                    "multilingual semantic similarity."
                )

            else:

                st.write(
                    "**Detected using Lebanese "
                    "Arabic civic keywords.**"
                )


with tab3:

    st.header(
        "🏛️ Municipality Dashboard"
    )

    st.write(
        "Monitor infrastructure reports and "
        "understand citizens' most common concerns."
    )

    total_reports = len(
        st.session_state.reports
    )

    total_suggestions = len(
        st.session_state.suggestions
    )

    if total_suggestions > 0:

        suggestion_topics = [
            item.get(
                "topic",
                "Unknown"
            )
            for item
            in st.session_state.suggestions
        ]

        topic_counts = pd.Series(
            suggestion_topics
        ).value_counts()

        top_priority = (
            topic_counts.index[0]
        )

    else:

        topic_counts = pd.Series(
            dtype=int
        )

        top_priority = (
            "No data yet"
        )

    col1, col2, col3 = st.columns(
        3
    )

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

    st.subheader(
        "💬 Citizen Feedback Analysis"
    )

    if total_suggestions == 0:

        st.info(
            "No citizen suggestions "
            "have been submitted yet."
        )

    else:

        st.write(
            "### Most Requested Topics"
        )

        topic_dataframe = (
            topic_counts
            .rename_axis(
                "Topic"
            )
            .reset_index(
                name="Number of Suggestions"
            )
        )

        st.dataframe(
            topic_dataframe,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            topic_dataframe.set_index(
                "Topic"
            )
        )

        st.success(
            f"🔥 Current top priority: "
            f"**{top_priority}**"
        )

        st.write(
            "### Municipality Feedback Summary"
        )

        (
            nlp_model,
            topic_names,
            topic_embeddings
        ) = load_nlp_resources()

        for topic, count in topic_counts.items():

            topic_suggestions = [
                item.get(
                    "suggestion",
                    ""
                )
                for item
                in st.session_state.suggestions
                if item.get(
                    "topic"
                ) == topic
            ]

            representative = (
                find_representative_feedback(
                    topic,
                    topic_suggestions,
                    nlp_model,
                    topic_names,
                    topic_embeddings
                )
            )

            with st.container(
                border=True
            ):

                st.write(
                    f"#### {topic}"
                )

                st.write(
                    f"**Citizen submissions:** "
                    f"{count}"
                )

                if count == 1:

                    st.write(
                        f"One citizen submission "
                        f"was grouped under "
                        f"**{topic.lower()}**."
                    )

                else:

                    st.write(
                        f"{count} citizen submissions "
                        f"were grouped under "
                        f"**{topic.lower()}**."
                    )

                st.write(
                    "**Representative concern:**"
                )

                st.write(
                    f"> {representative}"
                )

        st.write(
            "### Submitted Suggestions"
        )

        for item in reversed(
            st.session_state.suggestions
        ):

            with st.container(
                border=True
            ):

                topic = item.get(
                    "topic",
                    "Unknown"
                )

                suggestion_text = item.get(
                    "suggestion",
                    ""
                )

                score = item.get(
                    "match_score"
                )

                method = item.get(
                    "method",
                    "Previous semantic classification"
                )

                submitted_time = item.get(
                    "time",
                    "Earlier submission"
                )

                st.write(
                    f"**Topic:** "
                    f"{topic}"
                )

                st.write(
                    f"**Suggestion:** "
                    f"{suggestion_text}"
                )

                if score is not None:

                    st.caption(
                        f"Topic match score: "
                        f"{score:.1%} "
                        f"• {method} "
                        f"• Submitted: "
                        f"{submitted_time}"
                    )

                else:

                    st.caption(
                        f"{method} "
                        f"• Submitted: "
                        f"{submitted_time}"
                    )

    st.divider()

    st.subheader(
        "📷 Infrastructure Reports"
    )

    if total_reports == 0:

        st.info(
            "No infrastructure reports "
            "have been submitted yet."
        )

    else:

        for report in reversed(
            st.session_state.reports
        ):

            with st.container(
                border=True
            ):

                ticket_id = report.get(
                    "ticket_id",
                    "Unknown ticket"
                )

                issue = report.get(
                    "issue",
                    "Unknown"
                )

                location = report.get(
                    "location",
                    "Not specified"
                )

                note = report.get(
                    "note",
                    "No additional details"
                )

                confidence = report.get(
                    "confidence",
                    0
                )

                submitted_time = report.get(
                    "time",
                    "Earlier report"
                )

                status = report.get(
                    "status",
                    "Pending"
                )

                st.write(
                    f"### 🎫 {ticket_id}"
                )

                col_a, col_b = st.columns(
                    2
                )

                with col_a:

                    st.write(
                        f"**Issue:** "
                        f"{issue}"
                    )

                    st.write(
                        f"**Location:** "
                        f"{location}"
                    )

                    st.write(
                        f"**Citizen note:** "
                        f"{note}"
                    )

                with col_b:

                    st.write(
                        f"**AI confidence:** "
                        f"{confidence:.1%}"
                    )

                    st.write(
                        f"**Submitted:** "
                        f"{submitted_time}"
                    )

                status_icons = {
                    "Pending": "🟡",
                    "In Progress": "🔵",
                    "Resolved": "🟢"
                }

                st.write(
                    f"**Status:** "
                    f"{status_icons.get(status, '')} "
                    f"{status}"
                )

    st.divider()

    st.caption(
        "Nasij MVP — reports and suggestions "
        "are stored temporarily during the "
        "current application session."
    )
