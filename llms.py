## configure the llms module here and the embeddings
import os
import streamlit as st

# Check if we are running locally with a config file or in production
try:
    from config import set_environment
    set_environment()
except ImportError:
    # If config.py is missing (like on Streamlit Cloud), 
    # the app will look for GOOGLE_API_KEY in the environment/secrets automatically.
    pass

from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

chat_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
    max_tokens=None,
    max_retries=2,
    timeout=None
)

store = LocalFileStore('./cache/')
underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Avoiding multiple calls to the embedding model by caching them locally
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store, namespace=underlying_embeddings.model)