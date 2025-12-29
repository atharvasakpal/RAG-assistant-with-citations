# RAG Assistant with Citations

An intelligent Retrieval-Augmented Generation (RAG) assistant built using LangGraph, Gemini 1.5 Flash, and Streamlit. This system not only answers queries but verifies its own responses, provides citations, and maintains a persistent conversation history.

## Key Features

- **Document Intelligence**: Supports PDF, TXT, and DOCX files.  
- **Stateful Orchestration**: Managed by LangGraph for workflows: Retrieve → Generate → Verify → Finalize.  
- **Self-Correction**: Double-check node ensures answers are accurate, complete, and properly cited.  
- **Source Attribution**: Automatically cites specific document chunks (e.g., `[Source 1]`) for verification.  
- **Persistent Memory**: Maintains conversation context across multiple turns with LangGraph's MemorySaver.  
- **User-Friendly UI**: Streamlit interface with a document sidebar and expandable source views.  

## How It Works

The system operates as a state machine with the following nodes:

1. **Retrieve**: Searches the local ChromaDB vector store for relevant document context.  
2. **Generate**: Crafts an answer using Gemini 1.5 Flash with citations.  
3. **Double Check**: Reviews the draft answer for accuracy, completeness, and citation compliance.  
4. **Finalize**: Performs corrective rewriting if issues are found; otherwise delivers the verified answer.  

## Tech Stack

- **LLM**: Google Gemini 1.5 Flash  
- **Frameworks**: LangChain, LangGraph  
- **Vector Store**: ChromaDB  
- **Frontend**: Streamlit  
- **Embeddings**: Google Generative AI Embeddings (`gemini-embedding-001`)  

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Install Dependencies

Ensure Python 3.9+ is installed.
```bash
pip install -r requirements.txt
```
Note: Requirements should include pypdf, langchain-google-genai, langgraph, and langchain-chroma.

### 3. Set Up Environment Variables

Create a config.py file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```
### 4. Run the Application
```bash
streamlit run streamlit_app.py
```

## Project Structure
- **streamlit_app.py** — Web interface and file upload logic.

- **rag.py** — LangGraph definition and node orchestration.

- **retriever.py** — Document splitting and ChromaDB vector store management.

- **document_loader.py** — Specialized loaders for different file types.

- **llms.py** — Configuration for Gemini models and cached embeddings.

## Usage
- Upload supported documents (PDF, TXT, DOCX).

- Ask questions via the Streamlit interface.

- The assistant retrieves context, generates answers with citations, double-checks for accuracy, and delivers a final verified response.
