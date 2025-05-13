import streamlit as st
import pickle
import gensim
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the models
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
word2vec_model = pickle.load(open("word2vec_model.bin", "rb"))

# Preprocessing function (define this)
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Tokenize
    tokens = text.split()
    return tokens

# Function to preprocess and vectorize input text
def preprocess_and_vectorize(text):
    tokens = preprocess_text(text)
    word_vectors = []

    for word in tokens:
        if word in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[word])

    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(word2vec_model.vector_size)

# Streamlit app
st.title("Spam Classifier using Word2Vec and XGBoost")

# Input text
user_input = st.text_area("Enter the message to classify:")

if st.button("Classify"):
    if user_input.strip():
        vectorized_input = preprocess_and_vectorize(user_input)
        vectorized_input = vectorized_input.reshape(1, -1)

        # Predict using the model
        prediction = xgb_model.predict(vectorized_input)
        result = "Spam" if prediction[0] == 1 else "Not Spam"

        st.success(f"The message is classified as: {result}")
    else:
        st.error("Please enter a valid message.")
