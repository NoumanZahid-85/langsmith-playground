from typing import TypedDict, Annotated, Sequence 
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import os

load_dotenv()

# Initialize the LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5)

# Initialize Embeddings (Must use an actual embedding model, not a chat model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Loading the PDF
pdf_path = 'Stock_Market_Performance_2024.pdf'
try: 
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()
    print(f'PDF loaded: {len(pages)} pages')
except Exception as e:
    print(f'Error loading pdf: {e}')
    exit(1) 

# Chunking the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

persistent_directory = r'D:\Actual Portfolio Projects\langchain-langgraph-langsmith\05-langgraph-deep-dive'
collection_name = 'stock_performance_2024'

if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_name=collection_name
    )
    print(f'Vectorstore ready with {vectorstore._collection.count()} documents')
except Exception as e:
    print(f'Error creating vectorstore: {e}')
    exit(1)

# Creating a retriever
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

@tool
def retriver_tool(query: str) -> str:
    """Retrieve documents from the vectorstore Stock Market Performance 2024 based on the query"""
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f'Document {i+1}:\n{doc.page_content}')
    return "\n\n".join(results)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = [retriver_tool]
llm_with_tools = llm.bind_tools(tools)

def should_continue(state: AgentState):
    """Check if the last message contains any tool call"""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'continue'
    return 'end'

system_prompt = """
You are an AI assistant answering questions about Stock Market Performance in 2024 based on the provided PDF document.
Use the retriever tool to look up data. You can make multiple calls if needed.
Always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM Node
def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    # Add system prompt only if it's the first message
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# Tool Node
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    
    for t in tool_calls:
        print(f"\n[Searching database for: '{t['args'].get('query', '')}']")
        
        if t['name'] not in tools_dict:
            result = "Incorrect Tool Name."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    return {'messages': results}

# Build the Graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {"continue": "retriever_agent", "end": END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== STOCK MARKET RAG AGENT ===")
    
    while True:
        user_input = input("\nQuestion (or 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()