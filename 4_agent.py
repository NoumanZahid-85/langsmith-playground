from langchain_groq import ChatGroq          # <-- Groq instead of OpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()  # expects GROQ_API_KEY in .env

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'
    response = requests.get(url)
    return response.json()

# Use Groq instead of OpenAI
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # ✅ official replacement for deprecated models
    temperature=0,
)

# Pull the ReAct prompt from LangChain Hub (works with any LLM)
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# Test it
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)
print(response['output'])