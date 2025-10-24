# Revenue Planning AI Agent

**Agentic RAG-based assistant for revenue planning and marketing strategy**

## Features

-  **Agentic RAG Architecture** - Intelligent routing between knowledge base and web search
-  **Revenue Planning Expertise** - Specialized in CMO strategies, metrics (CAC, LTV, ARR, MQL), and budget optimization
-  **Real-time Web Search** - Powered by Tavily for current market data and trends
-  **Supabase Vector Store** - Fast similarity search with pgvector
-  **FastAPI Backend** - High-performance API with streaming support
-  **Cloud Run Deployment** - Auto-scaling serverless deployment with CI/CD
-  **CI/CD Pipeline** - Automated deployment on every push to main

## Architecture

This agent uses **Agentic RAG** with:
- **AI Brain** - Intelligently decides which tools to use based on the query
- **RAG Tool** - Retrieves from CMO Revenue Planning Playbook knowledge base
- **Tavily Tool** - Searches web for current information

## Deployment

Automatically deploys to Google Cloud Run via GitHub Actions on push to `main` branch.

**Service:** revenue-agent  
**Region:** us-central1

## Tech Stack

- **Framework:** FastAPI + LangGraph
- **LLM:** GPT-4o-mini (OpenAI)
- **Embeddings:** text-embedding-3-small
- **Vector Store:** Supabase (pgvector)
- **Search:** Tavily API
- **Deployment:** Google Cloud Run
- **CI/CD:** GitHub Actions

---
