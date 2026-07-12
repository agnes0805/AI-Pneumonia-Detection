import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import datetime

from gradcam import generate_gradcam
from report import generate_pdf_report
from chatbot import get_response
from utils import preprocess_image
from chatbot import get_response

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

body{
background-color:#f5f8fc;
}

.main-title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#0b5ed7;
margin-bottom:5px;
}

.sub-title{
font-size:18px;
text-align:center;
color:gray;
margin-bottom:25px;
}

.stButton>button{
width:100%;
border-radius:10px;
height:45px;
font-size:18px;
font-weight:bold;
background:#0b5ed7;
color:white;
}

.result-box{
padding:20px;
border-radius:15px;
background:#ffffff;
box-shadow:0px 4px 15px rgba(0,0,0,0.15);
margin-top:20px;
}

.prediction{
font-size:30px;
font-weight:bold;
text-align:center;
}

.normal{
color:green;
}

.pneumonia{
color:red;
}

.footer{
text-align:center;
color:gray;
margin-top:40px;
font-size:14px;
}

.chatbox{
padding:15px;
background:#eef6ff;
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_pneumonia_model.keras")
    return model

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
"""
<div class="main-title">
🫁 AI Pneumonia Detection System
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="sub-title">
Upload a Chest X-Ray Image to Detect Pneumonia using Deep Learning
</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/480/lungs.png",
        width=150
    )

    st.title("About")

    st.write(
        """
This AI system uses **DenseNet121**
to analyze Chest X-Ray images
and predict whether Pneumonia
is present.

Features:

✔ DenseNet121

✔ Grad-CAM

✔ Confidence Score

✔ PDF Report

✔ AI Chatbot
"""
    )

    st.info(
        "This application is for educational "
        "purposes only and is NOT a replacement "
        "for professional medical diagnosis."
    )

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

prediction = None
confidence = None
temp_path = None

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)
        temp_path = tmp.name

    processed = preprocess_image(temp_path)

    prediction_value = model.predict(processed)[0][0]

    if prediction_value > 0.55:

        prediction = "PNEUMONIA"

        confidence = prediction_value * 100

    else:

        prediction = "NORMAL"

        confidence = (1 - prediction_value) * 100
        # --------------------------------------------------
# DISPLAY PREDICTION
# --------------------------------------------------

    with col2:

        st.subheader("Prediction Result")

        if prediction == "PNEUMONIA":

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="prediction pneumonia">
                        ⚠️ {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="prediction normal">
                        ✅ {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("### Confidence Score")

        st.progress(float(confidence) / 100)

        st.success(f"{confidence:.2f}% Confidence")

        # --------------------------------------------------
        # MEDICAL RECOMMENDATION
        # --------------------------------------------------

        if prediction == "PNEUMONIA":
            st.markdown(
                """
                ### Medical Recommendation

                - This result suggests a high likelihood of pneumonia-like patterns on the X-ray.
                - Seek medical evaluation promptly.
                - A radiologist or physician should review the image and symptoms together.
                - Further tests such as a clinical exam, blood tests, or follow-up imaging may be required.
                - Do not start any treatment without professional medical advice.
                """
            )
        else:
            st.markdown(
                """
                ### Medical Recommendation

                - This X-ray appears more consistent with a normal chest radiograph.
                - If you have symptoms, still consult a doctor to confirm the diagnosis.
                - Continue routine health monitoring and seek care if symptoms develop.
                - A normal AI result is reassuring, but clinical context always matters.
                """
            )

    prediction_value = model.predict(processed)[0][0]

    st.write("Raw Prediction:", prediction_value)

    st.markdown("---")

# --------------------------------------------------
# GRAD-CAM VISUALIZATION
# --------------------------------------------------

    st.markdown("---")

    st.subheader("🧠 AI Attention (Grad-CAM)")

    try:

        heatmap = generate_gradcam(
            model,
            temp_path
        )

        grad_col1, grad_col2 = st.columns(2)

        with grad_col1:

            st.image(
                image,
                caption="Original Chest X-Ray",
                use_container_width=True
            )

        with grad_col2:

            st.image(
                heatmap,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

    except Exception as e:

        st.warning(
            f"Unable to generate Grad-CAM.\n\n{e}"
        )

# --------------------------------------------------
# REPORT GENERATION
# --------------------------------------------------

    st.markdown("---")

    st.subheader("📄 Medical Report")

    report = generate_pdf_report(
        prediction=prediction,
        confidence=confidence,
        image_path=temp_path
    )

    with open(report, "rb") as pdf:

        st.download_button(

            label="📥 Download PDF Report",

            data=pdf,

            file_name=f"Pneumonia_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",

            mime="application/pdf"

        )
        # --------------------------------------------------
# AI MEDICAL CHATBOT
# --------------------------------------------------

st.markdown("---")

st.subheader("💬 AI Medical Assistant")

st.markdown(
"""
<div class="chatbox">
Ask basic questions about pneumonia, symptoms,
prevention, treatment, or chest X-rays.
</div>
""",
unsafe_allow_html=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input(
    "Ask your question:",
    placeholder="Example: What are the symptoms of pneumonia?"
)

if st.button("Send"):

    if user_question.strip() != "":

        answer = get_response(user_question)

        st.session_state.chat_history.append(
            ("You", user_question)
        )

        st.session_state.chat_history.append(
            ("AI", answer)
        )

for sender, message in st.session_state.chat_history[::-1]:

    if sender == "You":

        st.markdown(
            f"""
            **🧑 You:** {message}
            """
        )

    else:

        st.markdown(
            f"""
            **🤖 AI:** {message}
            """
        )

# --------------------------------------------------
# MEDICAL DISCLAIMER
# --------------------------------------------------

st.markdown("---")

st.warning(
"""
### ⚠ Medical Disclaimer

This application is intended for educational and research
purposes only.

Predictions are generated using a Deep Learning model and
should **NOT** be considered a medical diagnosis.

Always consult a qualified radiologist or physician for
confirmation and treatment.
"""
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
"""
<div class="footer">

Developed using ❤️ Streamlit, TensorFlow,
DenseNet121 and Grad-CAM

© 2026 AI Pneumonia Detection System

</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# REMOVE TEMP FILE
# --------------------------------------------------

try:

    if temp_path is not None and os.path.exists(temp_path):

        os.remove(temp_path)

except Exception:

    pass
st.write("Raw prediction:", prediction_value)