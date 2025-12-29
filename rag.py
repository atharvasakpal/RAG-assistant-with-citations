from retriever import DocumentRetriever
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated
from llms import chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Modified system prompt to enforce citations
system_prompt = """You are a corporate assistant. Answer the question using ONLY the provided documents.
For every claim you make, you MUST cite the source using the format [Source X] where X is the document number.

Document Excerpts:
{context}

Final Answer Requirement:
- If the answer is found, include the citations.
- If not found, state that the documents do not contain the information.
"""

retriever = DocumentRetriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{question}'),
    ]
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    issues_report: str
    issues_detected: bool
    messages: Annotated[List, lambda x, y: x + y]

def retrieve(state: State):
    query = state['messages'][-1].content
    retrieved_docs = retriever.invoke(query)
    return {'context': retrieved_docs}

def generate(state: State):
    # Format context with source numbers for the LLM to reference
    docs_content = "\n\n".join([
        f"--- Source {i+1} (File: {doc.metadata.get('source', 'Unknown')}) ---\n{doc.page_content}" 
        for i, doc in enumerate(state["context"])
    ])
    
    messages = prompt.invoke({
        'question': state['messages'][-1].content, 
        'context': docs_content
    })
    
    response = chat_model.invoke(messages)
    return {'answer': response.content}

def double_check(state: State):
    # Check if citations are present if an answer was found
    check_prompt = (
        f"Does the following answer include citations like '[Source X]'? "
        f"If the answer says 'information not found', that is fine. "
        f"Otherwise, if citations are missing, return 'ISSUES FOUND: Missing Citations'.\n\n"
        f"Answer: {state['answer']}"
    )
    result = chat_model.invoke([{"role": "user", "content": check_prompt}])
    
    if "ISSUES FOUND" in result.content:
        return {"issues_report": result.content, "issues_detected": True}
    return {"issues_report": "", "issues_detected": False}

def doc_finalizer(state: State):
    if state.get("issues_detected"):
        # Re-run generation with emphasis on citations
        docs_content = "\n\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(state["context"])])
        revision_prompt = (
            f"Rewrite the answer including mandatory [Source X] citations.\n"
            f"Context: {docs_content}\n"
            f"Original Answer: {state['answer']}"
        )
        response = chat_model.invoke([{"role": "user", "content": revision_prompt}])
        return {"messages": [AIMessage(content=response.content)]}

    return {"messages": [AIMessage(content=state["answer"])]}

# Graph setup
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("double_check", double_check)
graph_builder.add_node("doc_finalizer", doc_finalizer)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "double_check")
graph_builder.add_edge("double_check", "doc_finalizer")
graph_builder.add_edge("doc_finalizer", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}