from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatGroq(model="llama-3.1-8b-instant", max_tokens=100)

def process(state: AgentState) -> AgentState:
    """This node will process the messages and return the response"""
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print("\nCurrent State:", state['messages'])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

conversation_history = []

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    conversation_history.append(HumanMessage(content=user_input))
    # Keep only the last 5 messages
    conversation_history = conversation_history[-5:]
    result = app.invoke({"messages": conversation_history})
    conversation_history = result['messages']

with open("chat_history.txt", "w") as f:
    f.write("Your Chat History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        else:
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of Conversation")
print("Conversation saved to chat_history.txt")
    