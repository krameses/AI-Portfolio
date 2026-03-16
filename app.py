import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Tools Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Tools Dashboard")

st.write("""
Welcome to the AI Tools App.

Use the **sidebar on the left** to navigate between tools.

Available tools:
- 🌍 Translator
- 📄 RAG Sample (Chat with Documents)
""")

st.divider()

st.subheader("About this app")

st.write("""
This app demonstrates multiple AI tools built using:

- Streamlit for UI
- LLM-based applications
- Retrieval-Augmented Generation (RAG)
""")

st.info("Select a page from the sidebar to get started.")