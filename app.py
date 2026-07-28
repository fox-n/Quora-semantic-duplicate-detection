"""
app.py

Streamlit app: semantic duplicate question checker.

Loads the fine-tuned Sentence Transformer model (Model 3, see
notebooks/04_sentence_transformers.ipynb) from models/sentence_transformer_duplicate_model,
encodes two user-entered questions, and classifies them as duplicate / not
duplicate using cosine similarity and the same threshold (0.60) chosen on
train and used in notebooks/06_error_analysis.ipynb and 07_test_evaluation.ipynb.
"""

import os

import streamlit as st
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentence_transformer_duplicate_model")
THRESHOLD = 0.60


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_PATH)


st.set_page_config(page_title="Quora Duplicate Question Checker")
st.image(os.path.join(os.path.dirname(__file__), "assets", "banner.png"))
st.title("Semantic Duplicate Question Detector")

st.write(
    "Enter two questions to check whether they are semantically "
    "equivalent (duplicates)."
)
st.caption("Note: the model was trained on English questions only -- enter questions in English.")

model = load_model()

question1 = st.text_input("Question 1")
question2 = st.text_input("Question 2")

if st.button("Check"):
    if not question1.strip() or not question2.strip():
        st.warning("Please enter both questions.")
    else:
        embeddings = model.encode([question1, question2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        similarity = max(0.0, min(1.0, similarity))

        is_duplicate = similarity > THRESHOLD

        if is_duplicate:
            st.success(f"Duplicate (similarity: {similarity:.3f})")
        else:
            st.info(f"Not a duplicate (similarity: {similarity:.3f})")

        st.caption(f"Decision threshold: {THRESHOLD}")
