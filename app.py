"""
app.py

Streamlit app: semantic duplicate question checker.

Planned UI:
- Two text inputs: Question 1, Question 2
- "Check" button
- Output: "Duplicate" / "Not duplicate" + confidence score

Model: Sentence Transformers (all-MiniLM-L6-v2), loaded from the trained
model produced in notebooks/04_sentence_transformers.ipynb.

TODO (week 7): implement model loading + inference + UI once the
Sentence Transformers model is trained and saved.
"""

import streamlit as st

st.set_page_config(page_title="Quora Duplicate Question Checker")
st.title("Semantic Duplicate Question Detector")

st.write(
    "Enter two questions to check whether they are semantically "
    "equivalent (duplicates)."
)

question1 = st.text_input("Question 1")
question2 = st.text_input("Question 2")

if st.button("Check"):
    st.warning("Model not yet integrated. Placeholder — coming in week 7.")
    # TODO: load Sentence Transformers model, compute embeddings,
    # cosine similarity, and classify as duplicate/not duplicate
    # with a confidence score.
