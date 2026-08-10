<div align="center">

# LangSmith Playground

**From prompt to production: LLM chains, RAG pipelines, autonomous agents, and stateful graphs.**
**All observable from a single dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6-FF6B35?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-00C7B7)](https://smith.langchain.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3-F55036)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What This Is

A hands-on reference that walks through five core LLM design patterns, each building on the last. Every run ships telemetry to **LangSmith**, so you can trace token flow, latency, and cost across the full stack without adding a single `print()`.

| #   | Module                                         | Pattern                 | Key Concept                                                                    |
| --- | ---------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| 01  | [LCEL Basics](01-lcel-basics/)                 | Prompt → LLM → Output   | Chain composition, sequential multi-step calls                                 |
| 02  | [RAG Pipeline](02-rag-pipeline/)               | Retrieve → Generate     | 4-version evolution from baseline to cached + fingerprinted                    |
| 03  | [ReAct Agent](03-react-agent/)                 | Thought → Act → Observe | Tool-calling agent with web search + REST APIs                                 |
| 04  | [LangGraph Evaluator](04-langgraph-evaluator/) | Fan-out → Merge → Score | Parallel state graph with structured Pydantic outputs                          |
| 05  | [LangGraph Deep Dive](05-langgraph-deep-dive/) | 9 interactive notebooks | Progressive patterns: basic → multi-input → sequential → conditional → looping |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LangSmith  Observability Layer                   │
│           traces · latency · token count · cost · debugging          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
 ┌──────────────┐      ┌───────────────┐      ┌────────────────┐
 │  LCEL Chains  │      │  ReAct Agent  │      │   LangGraph    │
 │  (01)         │      │  (03)         │      │   (04, 05)     │
 │               │      │               │      │                │
 │ Prompt → LLM  │      │ Reason → Act  │      │ State → Node   │
 │ → Parser      │      │ → Observe     │      │ → Edge → State │
 └──────┬────────┘      └───────┬───────┘      └────────────────┘
        │                       │
        ▼                       ▼
 ┌──────────────────────────────────────┐
 │         RAG Pipeline (02)            │
 │                                      │
 │  PDF → Chunk → Embed → FAISS Index   │
 │       ← Similarity Retrieve →        │
 │          → LLM Generate              │
 │                                      │
 │  v4: SHA-256 fingerprint caching     │
 └──────────────────────────────────────┘

 Models : Groq  → llama-3.3-70b-versatile
 Embeds : HuggingFace → sentence-transformers (local, free)
 Search : FAISS (in-memory vector similarity)
```

---

## Project Structure

```
langsmith-playground/
├── 01-lcel-basics/
│   ├── simple_llm_call.py            # Minimal LCEL chain
│   └── sequential_chain.py           # Multi-step chain with metadata tags
│
├── 02-rag-pipeline/
│   ├── rag_v1_baseline.py            # Standard chunk → embed → retrieve
│   ├── rag_v2_granular_tracing.py    # @traceable on each pipeline stage
│   ├── rag_v3_hierarchical_tracing.py # Single root run context
│   └── rag_v4_cached_fingerprint.py  # SHA-256 index caching on disk
│
├── 03-react-agent/
│   └── react_agent.py                # ReAct loop + DuckDuckGo + weather API
│
├── 04-langgraph-evaluator/
│   └── essay_evaluator.py            # Parallel fan-out StateGraph evaluator
│
├── 05-langgraph-deep-dive/
│   ├── 1-Basic_compliment_Agent.ipynb
│   ├── 2-Multi_Input_Agent-I.ipynb
│   ├── 3-Multi_Input_Agent-II.ipynb
│   ├── 4-Sequential_Graph-I.ipynb
│   ├── 4-Sequential_Graph-II.ipynb
│   ├── 5-Conditional-Agent-I.ipynb
│   ├── 5-Conditional-Agent-II.ipynb
│   ├── 6-Looping-Agent-I.ipynb
│   └── 6-Looping-Agent-II.ipynb
│
├── .env.example                      # API key template
├── requirements.txt                  # Pinned dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[Groq API Key](https://console.groq.com/)** (free tier available)
- **[LangSmith API Key](https://smith.langchain.com/)** (free tier available)

### Setup

```bash
# Clone
git clone https://github.com/NoumanZahid-85/langsmith-playground.git
cd langsmith-playground

# Virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Then fill in your GROQ_API_KEY and LANGCHAIN_API_KEY
```

### Run Any Module

```bash
# LCEL chain basics
python 01-lcel-basics/simple_llm_call.py
python 01-lcel-basics/sequential_chain.py

# RAG pipeline (needs a PDF named islr.pdf in the root)
python 02-rag-pipeline/rag_v4_cached_fingerprint.py

# ReAct agent with live web search
python 03-react-agent/react_agent.py

# LangGraph parallel essay evaluator
python 04-langgraph-evaluator/essay_evaluator.py
```

Open **[smith.langchain.com](https://smith.langchain.com)** to see every run traced in real time.

---

## Module Highlights

### RAG Evolution (4 versions)

| Version | What Changes                       | Why It Matters                              |
| ------- | ---------------------------------- | ------------------------------------------- |
| **v1**  | Baseline chunk + embed + retrieve  | Working RAG in 60 lines                     |
| **v2**  | `@traceable` on each step          | Granular LangSmith visibility               |
| **v3**  | Single root run wrapping all steps | Clean trace hierarchy                       |
| **v4**  | SHA-256 fingerprint + disk cache   | Skip re-indexing when source hasn't changed |

### LangGraph Evaluator

Parallel fan-out graph that scores an essay across three dimensions simultaneously (language, analysis, clarity), merges scores with a reducer, and synthesizes a final verdict with structured Pydantic outputs.

### LangGraph Deep Dive (9 Notebooks)

Progressive notebook series from a single-node compliment bot to multi-input agents, sequential pipelines, conditional branching, and looping game agents. Each notebook is self-contained and runnable.

---

## Tech Stack

| Layer         | Tool                                | Role                           |
| ------------- | ----------------------------------- | ------------------------------ |
| LLM           | Groq (`llama-3.3-70b-versatile`)    | Fast inference                 |
| Framework     | LangChain + LangGraph               | Chain/graph orchestration      |
| Embeddings    | HuggingFace `sentence-transformers` | Local, free vector embeddings  |
| Vector Store  | FAISS                               | In-memory similarity search    |
| Observability | LangSmith                           | Tracing, monitoring, debugging |
| Agent Tools   | DuckDuckGo Search, WeatherStack API | Real-time data retrieval       |

---

## License

MIT. Free to use, modify, and distribute.
