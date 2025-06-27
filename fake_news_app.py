# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 18:22:32 2025

@author: 91961
"""
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
# Ensure NLTK stopwords are available; this might be needed on a new deployment environment
# import nltk
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords') # If running on a fresh environment, uncomment and add other necessary downloads


# --- Configuration ---
# Set page title and favicon
st.set_page_config(page_title="Fake News Classifier", page_icon="📰")

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the loading of these heavy objects
def load_resources():
    try:
        with open('logistic_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Make sure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

model, tfidf_vectorizer = load_resources()

# --- Text Cleaning Function (must be identical to training) ---
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])
    return text

# --- Streamlit UI ---
st.title("📰 Fake News Classifier")
st.markdown("Enter a news article (title + text) below to determine if it's fake or real.")

# Text input area
user_input = st.text_area("Enter News Article Here:", height=300,
                          placeholder="Paste the full news article (title and body) here...")

if st.button("Classify"):
    if user_input:
        # Preprocess the user input
        cleaned_input = clean_text(user_input)

        # Transform input using the loaded TF-IDF Vectorizer
        # The input needs to be in a list format for the vectorizer
        input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        # Make prediction
        prediction = model.predict(input_tfidf)
        prediction_proba = model.predict_proba(input_tfidf)

        st.subheader("Prediction:")
        if prediction[0] == 0:
            st.error("🚨 This news article is likely **FAKE**.")
            st.markdown(f"**Confidence:** {prediction_proba[0][0]*100:.2f}% (Fake) | {prediction_proba[0][1]*100:.2f}% (True)")
        else:
            st.success("✅ This news article is likely **TRUE**.")
            st.markdown(f"**Confidence:** {prediction_proba[0][1]*100:.2f}% (True) | {prediction_proba[0][0]*100:.2f}% (Fake)")

        st.markdown("---")
        st.subheader("Explanation:")
        st.info("""
            This prediction is based on a Logistic Regression model trained on a large dataset of fake and real news.
            The model analyzes patterns of words and their importance (TF-IDF scores) within the article
            to classify it. While the model shows high accuracy, no model is perfect.
            Always cross-reference information with reliable sources.
        """)
    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")
st.markdown("Built as part of the AIML Internship Project.")
