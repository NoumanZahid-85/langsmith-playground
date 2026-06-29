from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
os.environ['LANGCHAIN_PROJECT'] = "langchain-part2"
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config={
    'run_name':'Sequential-Chain-Test-1',
    'tags':['sequential-chain', 'part2', 'langsmith'],
    'metadata':{
        'purpose': 'testing langsmith tracing',
        'source_file': '2_sequential_chain.py'
    }
}

result = chain.invoke({'topic': 'Unemployment in Pakistan'}, config=config)

print(result)
