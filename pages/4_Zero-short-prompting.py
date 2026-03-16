from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
from langchain_groq import ChatGroq
# Get API key securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")    
model_name = "llama-3.1-8b-instant"
# Initialize the chatbot
chat_model = ChatGroq(model_name=model_name, groq_api_key=GROQ_API_KEY)

input_text = input("Type the word or sentence: ")
zero_shot_prompt=f"Translate the following input to tamil: {input_text}"
response_zero_short = chat_model.invoke(zero_shot_prompt)
print("Response from the model:\n", response_zero_short.content)