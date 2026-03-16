import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS

st.title("English → Hindi Translator")

english_text = st.text_input("Enter English sentence")

if st.button("Translate and Speak"):

    # translate
    hindi_text = GoogleTranslator(source='en', target='hi').translate(english_text)

    st.subheader("Hindi Translation")
    st.write(hindi_text)

    # text to speech
    tts = gTTS(text=hindi_text, lang='hi')
    tts.save("speech.mp3")

    with open("speech.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/mp3")