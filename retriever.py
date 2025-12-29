import os
from typing import List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from document_loader import load_document
from llms import EMBEDDINGS

# persistent ChromaDB
VECTOR_STORE = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDINGS)

def split_documents(docs: List[Document]) -> list[Document]:
    if not docs:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(docs)

class DocumentRetriever(BaseRetriever):
    documents: List[Document] = []
    k: int = 5

    def model_post_init(self, ctx: Any) -> None:
        if self.documents:
            self.store_documents(self.documents)

    @staticmethod
    def store_documents(docs: List[Document]) -> None:
        splits = split_documents(docs)
        if splits:
            VECTOR_STORE.add_documents(splits)

    def add_uploaded_docs(self, uploaded_files):
        docs = []
        for file in uploaded_files:
            temp_filepath = f"./temp_upload_{file.name}"
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            try:
                loaded_docs = load_document(temp_filepath)
                if loaded_docs:
                    docs.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading document {file.name}: {e}")
            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)

        if docs:
            self.documents.extend(docs)
            self.store_documents(docs)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    # This remains the same as the previous fix
        return VECTOR_STORE.similarity_search(query=query, k=self.k)