# LangSmith Playground

A practical reference repository demonstrating LLM application development, Retrieval-Augmented Generation (RAG), autonomous agents, and stateful workflows using LangChain, LangGraph, and Groq, with full observability powered by LangSmith.

---

## Overview

This repository provides step-by-step implementations showing how to build and monitor production-grade LLM pipelines. Each module illustrates a core architectural pattern, moving from basic prompt chains to advanced cached RAG systems and multi-agent graph evaluations.

All models run using Groq's high-throughput API (`llama-3.3-70b-versatile`), paired with open-source local embeddings from HuggingFace (`sentence-transformers`) for fast, local vector search.

---

## Core Modules

### 1. Simple LLM Call (`1_simple_llm_call.py`)
- Demonstrates basic LangChain Expression Language (LCEL) constructs.
- Connects `PromptTemplate`, `ChatGroq`, and `StrOutputParser` into a seamless chain.

### 2. Sequential Chain (`2_sequential_chain.py`)
- Links multiple LLM calls sequentially, passing the output of a detailed topic generation directly into a summary generator.
- Attaches custom metadata, tags, and run names to LangSmith for targeted filtering in the dashboard.

### 3. RAG Evolution (`3_rag_v1.py` through `3_rag_v4.py`)
Four progressive iterations of Document RAG built over a PDF document (`islr.pdf`):
- **v1 (Baseline)**: Standard chunking with `RecursiveCharacterTextSplitter`, local embeddings via HuggingFace, and similarity search with FAISS.
- **v2 (Granular Tracing)**: Adds LangSmith `@traceable` annotations to document loading, text splitting, and vectorstore creation steps.
- **v3 (Hierarchical Tracing)**: Wraps pipeline setup and query execution under a single root run context (`pdf_rag_full_run`) to simplify trace viewing.
- **v4 (Production Caching & Fingerprinting)**: Introduces local disk index caching (`.indices`) driven by SHA-256 fingerprinting of document bytes, chunk size, and embedding model parameters. Bypasses re-indexing when source files are unchanged.

### 4. ReAct Autonomous Agent (`4_agent.py`)
- Implements a Reasoning and Acting (ReAct) loop using LangChain's `create_react_agent` and `AgentExecutor`.
- Integrates web search tools (`DuckDuckGoSearchRun`) and custom REST API integration tools (`get_weather_data`) to solve complex multi-step queries.

### 5. Multi-Criteria LangGraph Evaluator (`5_langgraph.py`)
- Builds a parallel execution StateGraph using LangGraph to evaluate essay quality across three distinct dimensions simultaneously: language quality, depth of analysis, and clarity of thought.
- Enforces structured JSON outputs using Pydantic schemas (`EvaluationSchema`), merges scores dynamically using reducer operators, and synthesizes a final score and feedback summary.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Groq API Key
- LangSmith API Key (for tracing and telemetry)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NoumanZahid-85/langsmith-playground.git
cd langsmith-playground
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements_fixed.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory by copying the example template:
```bash
cp .env.example .env
```
Populate `.env` with your API keys:
```ini
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=langsmith-playground
```

---

## Running the Examples

Execute any script directly using Python:

```bash
# Basic LCEL chain
python 1_simple_llm_call.py

# Sequential chain with tracing
python 2_sequential_chain.py

# Production cached RAG
python 3_rag_v4.py

# ReAct Agent
python 4_agent.py

# LangGraph essay evaluation
python 5_langgraph.py
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                        LangSmith Observability                    │
│              (Tracing, Monitoring, Debugging)                     │
└───────────────────────────────────────────────────────────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     │                            │                            │
     ▼                            ▼                            ▼
┌─────────────┐          ┌────────────────┐          ┌────────────────┐
│  LCEL Chain │          │   ReAct Agent  │          │   LangGraph    │
│ (1, 2)      │          │   (4)          │          │   (5)          │
│             │          │                │          │                │
│ Prompt →    │          │ Thought →      │          │ State → Node → │
│ LLM →       │          │ Action →       │          │ Node → State   │
│ Output      │          │ Observation    │          │                │
└─────────────┘          └────────────────┘          └────────────────┘
       │                        │
       ▼                        ▼
┌─────────────────────────────────────┐
│         RAG Pipeline (3)            │
│                                     │
│ PDF → Chunk → Embed → VectorStore   │
│        ← Retrieve → Generate        │
└─────────────────────────────────────┘
```

---

## License

MIT License. Free to use and modify for personal or commercial projects.
