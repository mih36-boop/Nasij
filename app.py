import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


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

#header

st.title("🧶 Nasij")
st.subheader("Weaving communities together through AI")

st.write(
    "Nasij helps citizens report infrastructure issues "
    "and share suggestions with municipalities."
)


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


#tab2 nlp

with tab2:

    st.header("Submit a Suggestion")

    st.write(
        "Share an idea or concern with your municipality."
    )


#tab3 dashboard

with tab3:

    st.header("Municipality Dashboard")

    st.write(
        "View citizen reports and the most common concerns."
    )
