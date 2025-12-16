import streamlit as st
import pickle
# import speech_recognition as sr
import numpy as np

# Page config

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# Custom CSS

st.markdown("""
<style>
.main {
    background-color: Light Blue;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 22px;
    text-align: center;
    margin-top: 15px;
}
button {
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# Load trained model & vectorizer

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# App Header

st.title("💬 Sentiment Analysis App")
st.caption(" Built with Python + AI Project")

st.write("Analyze emotions from text or voice input")

# Voice input function

# def voice_input():
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("🎤 Speak now...")
#         r.adjust_for_ambient_noise(source, duration=0.5)
#         audio = r.listen(source)

#         try:
#             text = r.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             return ""
#         except sr.RequestError:
#             st.error("Speech recognition service error")
#             return ""

# Result display function (WITH CONFIDENCE)

def show_result(result, confidence):
    if result == "positive":
        bg = "#d4edda"
        emoji = "😊"
    elif result == "negative":
        bg = "#f8d7da"
        emoji = "😠"
    else:
        bg = "#fff3cd"
        emoji = "😐"

    st.markdown(
        f"""
        <div class='result-box' style='background:{bg};'>
            {emoji} <b>{result.capitalize()} Sentiment</b><br>
            🔍 Confidence: <b>{confidence:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# Suggestion inputs

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("✍️ Type your text here")

# # Voice input button

# st.write("---")
# if st.button("🎤 Voice Input"):
#     spoken_text = voice_input()

#     if spoken_text:
#         st.success(f"🗣️ You said: {spoken_text}")
#         vector = vectorizer.transform([spoken_text.lower()])
#         result = model.predict(vector)[0]
#         confidence = np.max(model.predict_proba(vector)) * 100
#         show_result(result, confidence)
#     else:
#         st.warning("Voice not recognized. Please try again.")


with col2:
    st.write("🎯 **Inputs Like**")
    st.write("- I am very happy")
    st.write("- This is terrible")
    st.write("- It is okay")

# Analyze text button

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text = user_input.lower()
        vector = vectorizer.transform([text])
        result = model.predict(vector)[0]
        confidence = np.max(model.predict_proba(vector)) * 100
        show_result(result, confidence)


# Footer

st.write("---")

st.caption("🚀 AI Project with Streamlit + Machine Learning")
