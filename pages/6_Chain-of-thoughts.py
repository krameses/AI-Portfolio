from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

input_text = """
In a charming village, siblings Jack, Jill and Tom set out on a quest to fetch water from a hilltop well.
As they climbed, singing joyfully, misfortune struck—Jack tripped on a stone and tumbled down the hill,
with Jill following suit. Though slightly battered, the pair returned home to comforting embraces.
"""

prompt_2 = """
Your task is to perform the following actions:

1 - Summarize the following text delimited by <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a JSON object that contains the following keys: french_summary, num_names.

Text: <{text}>
"""

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq model
chat_model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_2
)

# Create chain (NEW LangChain way)
chain = prompt | chat_model

# Run chain
response = chain.invoke({"text": input_text})

print(response.content)