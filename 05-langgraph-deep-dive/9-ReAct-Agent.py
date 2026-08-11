from typing import TypedDict, Annotated, Sequence # TypedDict is used to define the state of the graph, Annotated is used to define the type of the state, Sequence is used to define the type of the messages
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage,HumanMessage, ToolMessage # BaseMessage is the base class for all messages, SystemMessage is used to define the system message, ToolMessage is used to define the tool message
from langchain_groq import ChatGroq
from langchain_core.tools import tool # tool is used to define the tools
from langgraph.graph.message import add_messages  # add_messages is used to add messages to the state
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode # ToolNode is used to wrap the tools

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""  # Without this docstring, the LLM will not know what to do with the tool
    return f"At this time, the weather in {city} seems to be cloudy with a high of 75 degrees Fahrenheit and a low of 60 degrees Fahrenheit."

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [get_weather, add_numbers, subtract_numbers, multiply_numbers]

llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)

def call_llm(state: AgentState) -> AgentState:
    """This node will call the LLM and return the response"""
    system_prompt = SystemMessage(content="You are a helpful assistant. You must answer ALL parts of the user's question. If you use multiple tools, ensure you summarize the results of EVERY tool call in your final response.")
    messages_to_send = [system_prompt] + state['messages']
    response = llm.invoke(messages_to_send)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """This node will decide whether to continue the graph or not"""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"
tool_node = ToolNode(tools)
graph = StateGraph(AgentState)
graph.add_node("call_llm", call_llm)
graph.add_node("tools", tool_node)
graph.add_edge(START, "call_llm")
# Conditional node only provide one-way edge either tools or END, but not the reverse edge
graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
# This edge is to provide the reverse edge from tools to call_llm
graph.add_edge("tools", "call_llm")
app = graph.compile()

def print_stream(stream):
    for update in stream:
        # Loop through each node's update
        for node_name, state_update in update.items():
            # Print every message generated in this step
            for message in state_update.get("messages", []):
                message.pretty_print()


inputs = {"messages": [HumanMessage(content="What is 3 plus 7 and multiply answer with 8 and also what is the weather in New York?")]}
print_stream(app.stream(inputs, stream_mode="updates"))