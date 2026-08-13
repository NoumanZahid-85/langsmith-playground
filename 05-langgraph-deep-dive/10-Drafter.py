from typing import TypedDict, Annotated, Sequence 
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Update the document with the given content"""
    global document_content
    document_content = content 
    
    # --- NEW ADDITION: Print the draft to the terminal ---
    print("\n" + "="*40)
    print("📄 CURRENT DOCUMENT DRAFT:")
    print("="*40)
    print(document_content)
    print("="*40 + "\n")
    
    return "Document updated successfully."

@tool
def save(filename: str) -> str:
    """Save the current document to the text file and finish the process."""
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"
    try:
        with open(filename, "w") as f:
            f.write(document_content)
        return f"Document saved successfully as {filename}"
    except Exception as e:
        return f"Error saving document: {e}"

tools = [update, save]
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.5).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    
    The current document content is: {document_content}""")
    
    # FIX 2: Clean terminal greeting and input handling
    if not state["messages"]:
        print("AI: I am ready to help you draft your document. What would you like to work on?")
        
    user_input = input("\nUser: ")
    user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = llm.invoke(all_messages)
    
    # Cleanly print AI response
    if response.content:
        print(f"AI: {response.content}")
        
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[AI is using tools: {[tc['name'] for tc in response.tool_calls]}]")
        
    return {'messages': list(state['messages']) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""
    messages = state['messages']
    if not messages:
        return 'continue'
        
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            'saved' in message.content.lower() and 
            'document' in message.content.lower()):
            return 'end' 
    return 'continue' 

# We removed the messy print_messages function entirely!

graph = StateGraph(AgentState)
graph.add_node('agent', our_agent)
graph.add_node('tools', ToolNode(tools))
graph.set_entry_point('agent')
graph.add_edge('agent', 'tools')
graph.add_conditional_edges(
    'tools',
    should_continue,
    {
        'continue': 'agent',
        'end': END
    }
)
app = graph.compile()

def run_document_agent():
    print("\n========= Drafter Agent ========")
    state = {'messages': []}
    
    # Stream silently, all prints are handled inside the nodes now
    for s in app.stream(state, stream_mode="values"):
        pass 
        
    print("\n===== Drafter Finished ======\n")

if __name__ == "__main__":
    run_document_agent()