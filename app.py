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
        body, .stApp {
            background-color: #ffe6f0;
        }
        .big-font {
            font-size:30px !important;
            color: #ff66a3;
            text-align: center;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #ff99cc;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            height: 3em;
            width: 100%;
        }
        .stTextInput>div>div>input {
            background-color: #fff0f5;
        }
        .stTextArea>div>textarea {
            background-color: #fff0f5;
        }
        .stRadio>div>label {
            color: #cc3366;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🎀 UI
st.markdown('<p class="big-font">🎀 Satisfactory Prediction App 🎀</p>', unsafe_allow_html=True)

# 💌 Minta nama user
name = st.text_input("Siapa namamu, bestie? 🥰")

st.success(f"Halo {name}! ✨ Yuk, masukkan ulasanmu di bawah ini 💖")


st.title("Satisfactory Prediction")

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
    st.success(f"**Sentiment Analysis:** {sentiment}")
    st.stop()
