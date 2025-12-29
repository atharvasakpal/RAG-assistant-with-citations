## configure the llms module here and the embeddings
from config import set_environment
set_environment()

from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

chat_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash', # Updated from 2.5 to 1.5
    temperature=0,
    max_tokens=None,
    max_retries=2,
    timeout=None
)

store = LocalFileStore('./cache/')
underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#avoiding multiple calls to the embedding model by caching them locally
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings,store,namespace=underlying_embeddings.model)