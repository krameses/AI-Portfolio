import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.title("📊 Excel AI Agent")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:

    # Read Excel
    df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # Create dataframe agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="tool-calling",
        allow_dangerous_code=True
    )

    # User question
    question = st.text_input("Ask a question about your data")

    if question:
        with st.spinner("Analyzing data..."):
            response = agent.invoke(question)
            st.write(response["output"])