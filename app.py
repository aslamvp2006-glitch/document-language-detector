import streamlit as st
import joblib
from pathlib import Path
from pypdf import PdfReader
from docx import Document
model = joblib.load("document_language_detector.pkl")

# Page configuration
st.set_page_config(
    page_title="Document Language Detector",
    page_icon="🌐",
    layout="centered"
)


# Load trained machine-learning model
@st.cache_resource
def load_model():
    return joblib.load("document_language_detector.pkl")


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# Extract text from uploaded documents
def extract_text(uploaded_file):

    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".txt":

        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

    elif extension == ".pdf":

        reader = PdfReader(uploaded_file)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    elif extension == ".docx":

        document = Document(uploaded_file)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    return ""


# Website heading
st.title("🌐 Document Language Detector")

st.write(
    "Identify the language of text documents "
    "using Machine Learning and Natural Language Processing."
)

st.divider()


# Text input
st.subheader("✍️ Enter Text")

text_input = st.text_area(
    "Type a sentence or paragraph:",
    placeholder="Example: Hello, how are you?",
    height=150
)


# Document upload
st.subheader("📄 Upload a Document")

uploaded_file = st.file_uploader(
    "Choose a TXT, PDF, or DOCX file",
    type=["txt", "pdf", "docx"]
)


# Prediction button
if st.button(
    "🔍 Detect Language",
    type="primary",
    use_container_width=True
):

    try:

        if uploaded_file is not None:

            content = extract_text(uploaded_file)

        else:

            content = text_input

        if not content or not content.strip():

            st.warning(
                "Please enter text or upload a document."
            )

        else:

            with st.spinner("Detecting language..."):

                prediction = model.predict([content])[0]

            st.success("Language detected successfully!")

            st.subheader("Prediction Result")

            st.info(f"🌍 Detected Language: {prediction}")

    except Exception as e:

        st.error(f"An error occurred: {e}")


st.divider()

st.caption(
    "Mini Project | Machine Learning | Natural Language Processing"
)
