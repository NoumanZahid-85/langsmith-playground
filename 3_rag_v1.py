from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings   # Free, local embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq   # <--- Groq LLM
import os
os.environ['LANGCHAIN_PROJECT'] = "Rag Tracing V1"
load_dotenv()  # expects GROQ_API_KEY in .env
PDF_PATH = "islr.pdf"  

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index using FREE local HuggingFace embeddings
# "all-MiniLM-L6-v2" is fast and good for retrieval (~384 dims)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain with Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
while True:
    q = input("\nQ: ")
    if q.lower() in ("exit", "quit"):
        break
    ans = chain.invoke(q.strip())
    print("\nA:", ans)