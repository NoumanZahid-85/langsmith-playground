from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Pakistan?"})
print(result)
