import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------------------
# Load environment variables
# ---------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("📄 Chat with your PDF")

# ---------------------------
# Upload PDF
# ---------------------------

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# ---------------------------
# Cache vector store
# ---------------------------

@st.cache_resource
def create_vectorstore(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# ---------------------------
# If PDF uploaded
# ---------------------------

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    vectorstore = create_vectorstore("temp.pdf")
    retriever = vectorstore.as_retriever()

    # ---------------------------
    # LLM
    # ---------------------------

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    # ---------------------------
    # Chat history
    # ---------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask something about the PDF")

    if prompt:

        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        response = qa_chain.invoke({"question": prompt})

        answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })