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
        .stApp {
            background: linear-gradient(to bottom, #ffffff, #fdefff);
        }
        html, body, [class*="css"] {
            color: black !important;
            font-family: "Arial", sans-serif;
        }
        .big-font {
            font-size:30px !important;
            color: #ff66a3;
            text-align: center;
            font-weight: bold;
        }
        .stButton>button {
            background-color: black !important;
            color: white !important;
            border-radius: 10px;
            font-size: 16px;
            height: 3em;
            width: 100%;
            border: none;
        }
        .stTextInput>div>div>input {
            background-color: black !important;
            border: 1px solid #ffb6c1 !important;
            color: white !important;
        }
        .stTextArea>div>textarea {
            background-color: #fff0f5 !important;
            border: 1px solid #ffb6c1 !important;
            color: black !important;
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #cc3366 !important;
        }
        label, .stTextInput label, .stTextArea label, .stRadio label {
            color: black !important;
            font-weight: bold;
        }
        /* Sentiment and Hello Message Color */
        .highlight {
            color: #ff66a3;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)


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
