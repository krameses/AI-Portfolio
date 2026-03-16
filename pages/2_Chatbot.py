from dotenv import load_dotenv
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize model
model_name = "llama-3.1-8b-instant"
chat_model = ChatGroq(model_name=model_name, groq_api_key=GROQ_API_KEY)

st.set_page_config(page_title="My Chatbot", page_icon="")

st.title("My Chatbot UI")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to LLM
    response = chat_model.invoke([HumanMessage(content=prompt)])

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)