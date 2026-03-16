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
input_prompt = input('type the paragraph to summarize : ')
few_shot_prompt = f"""
                    You are a summarization assistant. Summarize the following paragraphs into one concise sentence. Only return the summary, and do not repeat the input.

                    Text: Climate change is one of the most significant global challenges today. It leads to rising temperatures, extreme weather events, and threatens ecosystems and human health.
                    Summary: Climate change causes severe environmental and health problems worldwide.

                    Text: Artificial Intelligence is transforming industries by automating tasks, analyzing data, and enabling smarter decisions across healthcare, finance, and more.
                    Summary: AI is revolutionizing industries through automation and data-driven decision making.

                    Text: {input_prompt}
                    Summary:"""
response_few_short = chat_model.invoke(few_shot_prompt)
print("Response from the model:\n", response_few_short.content)