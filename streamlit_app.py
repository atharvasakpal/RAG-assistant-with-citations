__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from langchain_core.messages import HumanMessage
from document_loader import DocumentLoader
from rag import graph, config, retriever

# 1. Page Configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")

try:
    from config import set_environment
    set_environment()
except ImportError:
    # If config.py is missing (e.g., in production), 
    # Streamlit will automatically check st.secrets for environment variables.
    pass

# 2. Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# 3. Sidebar: Document Management
with st.sidebar:
    st.title("📂 Document Center")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=list(DocumentLoader.supported_extensions.keys()),
        accept_multiple_files=True
    )
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
        if new_files:
            with st.spinner("Indexing new documents..."):
                # Add only new files to the retriever/vector store
                retriever.add_uploaded_docs(new_files)
                for f in new_files:
                    st.session_state.uploaded_files.append(f.name)
            st.success(f"Indexed {len(new_files)} new files!")

    if st.session_state.uploaded_files:
        st.write("### Indexed Files:")
        for file_name in st.session_state.uploaded_files:
            st.caption(f"✅ {file_name}")

# 4. Main Chat Interface
st.title("🤖 RAG Assistant with Citations")
st.markdown("Ask questions about your uploaded documents. The AI will cite its sources.")

# Display chat messages from history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Logic
if user_input := st.chat_input("How can I help you today?"):
    # Display user message
    with st.chat_message("User"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "User", "content": user_input})

    # Generate response via LangGraph
    with st.chat_message("Assistant"):
        with st.spinner("Searching and thinking..."):
            # Invoke the graph
            # Note: config provides the thread_id for memory persistence
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )
            
            answer = result["messages"][-1].content
            sources = result.get("context", [])
            
            st.markdown(answer)
            
            # 6. Display Citations in Expander
            if sources:
                with st.expander("📚 View Citations"):
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get('source', 'Unknown source')
                        # Clean up the temp path if it exists in metadata
                        clean_name = os.path.basename(source_name).replace("temp_upload_", "")
                        st.write(f"**[Source {i+1}]** - {clean_name}")
                        st.caption(doc.page_content[:300] + "...")
            
            st.session_state.chat_history.append({"role": "Assistant", "content": answer})