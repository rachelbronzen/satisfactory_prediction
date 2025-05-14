import streamlit as st
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from lexicon_sentiment import get_sentiment

cnn_model = load_model("cnn_model.h5")
logistic_model = joblib.load("logistic_regression_model.pkl")


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vectorizer = joblib.load("vectorizer.pkl")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


#STREAMLIT APP
st.markdown(
"""
    <style>
        /* GLOBAL STYLES */
        body, .stApp {
            background: linear-gradient(to bottom, #04031a, #0d0b38);
            color: black;
        }

        /* TITLE */
        .big-font {
            font-size: 40px;
            color: #efa0cd;
            font-weight: bold;
            text-align: center;
        }

        /* SUBTITLE */
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #f5cae2;
        }

        /* TEXT INPUT, TEXT AREA, RADIO, BUTTON */
        input[type="text"],
        textarea,
        .stTextInput input,
        .stTextArea textarea,
        .stButton > button {
            background-color: #ffe3f3 !important;
            color: #e87fa3 !important;
            border: none;
        }

        /* Make radio text pink too */
        .stRadio label {
            color: #e87fa3 !important;
        }

        /* Button hover effect */
        .stButton > button:hover {
            background-color: #f8cde0 !important;
            color: #9b1859 !important;
        }

        /* Placeholder text */
        ::placeholder {
            color: #e87fa3 !important;
            opacity: 1;
        }

        /* Message boxes (like st.warning, st.success) */
        .stAlert {
            color: black;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<p class="big-font">Satisfactory Prediction</p>', unsafe_allow_html=True)

name = st.text_input("Hello! What is your name? 👋", placeholder="Your Name")

st.markdown(f'<div style="color:#ff66a3; font-weight:bold;">Hello {name}! Thank you for using our app. Please enter your review below:</div>', unsafe_allow_html=True)

user_input = st.text_area("Review:")
model_choice = st.radio("Choose Model:", ["CNN", "Logistic Regression"])

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter your review!!!")
    else:
        if model_choice == "CNN":
            seq = tokenizer.texts_to_sequences([user_input])
            pad = pad_sequences(seq, maxlen=200)
            pred = cnn_model.predict(pad)
            score_index = np.argmax(pred)
            
        
        elif model_choice == "Logistic Regression":
            vec = vectorizer.transform([user_input])
            pred = logistic_model.predict_proba(vec)
            score_index = np.argmax(pred)
    
    score = label_encoder.inverse_transform([score_index])[0]
    st.write(f"**Score Prediction {model_choice} :** {score}")
    sentiment = get_sentiment(user_input)
    st.markdown(f'<div style="color:#ff66a3; font-weight:bold;">Sentiment Analysis: {sentiment}</div>', unsafe_allow_html=True)
    st.stop()
