import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline using the model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = load_model()

# Streamlit app UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review or any text, and the model will predict sentiment (1 to 5 stars).")

# User input
user_input = st.text_area("Enter your review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = sentiment_pipeline(user_input)
            label = result[0]['label']
            score = result[0]['score']

            # Interpretation
            star = label.split()[0]  # Extract "1", "2", ... from "1 star", "5 stars", etc.
            st.success(f"⭐ Sentiment: **{label}** ({round(score * 100, 2)}% confidence)")

            # Optional: Text-based interpretation
            interpretation = {
                "1": "Very Negative 😠",
                "2": "Negative 🙁",
                "3": "Neutral 😐",
                "4": "Positive 🙂",
                "5": "Very Positive 😄"
            }
            st.markdown(f"### Interpretation: **{interpretation.get(star, 'Unknown')}**")
