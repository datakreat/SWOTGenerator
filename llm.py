import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI


#Langchain-groq
# def generate_response(prompt):
#     load_dotenv()
#     api = os.getenv('groq_api_key')
#     # Initialize the LLM with the provided API key and model name
#     llm = ChatGroq(temperature=0.5, groq_api_key=api, model="llama3-8b-8192")
#     response = llm.invoke(prompt)
#     return response.content

#azure-openai
def generate_response(prompt):
    load_dotenv()
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION')

    # Initialize the Azure OpenAI LLM
    llm = AzureChatOpenAI(
        openai_api_key=api_key,
        azure_deployment=deployment_name,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        temperature=0.5
    )

    # Generate a response from the prompt
    response = llm.invoke(prompt)

    return response.content

# print(generate_response('Capital of India?'))