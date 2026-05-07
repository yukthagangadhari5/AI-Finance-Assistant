# AI Finance Assistant 🏦

An intelligent multi-agent RAG system that answers financial questions 
using real RBI (Reserve Bank of India) annual reports.

## 🚀 Demo
[Add screenshot here]

## 🏗️ Architecture
- **Multi-Agent System** (LangGraph): Router Agent → Research Agent
- **RAG Pipeline**: 9,189 chunks indexed from 24 RBI documents
- **Vector Search**: FAISS with HuggingFace embeddings (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 1.5 Flash via Google AI Studio API
- **Backend**: FastAPI with OpenAPI schema
- **Frontend**: React + Vite

## 📊 Performance
- Retrieval latency: ~86ms
- Average confidence score: 0.63
- Documents indexed: 24 RBI Annual Reports (2022-2025)
- Total chunks: 9,189

## 🛠️ Tech Stack
- Python, FastAPI, LangChain, LangGraph, FAISS
- Google Gemini API, HuggingFace Transformers
- React, Vite
- PostgreSQL, MongoDB

## ⚙️ Setup

### Backend
pip install -r requirements.txt
python backend/ingest.py
uvicorn backend.main:app --reload

### Frontend
cd frontend
npm install
npm run dev

## 🔑 Environment Variables
Create a .env file:
GOOGLE_API_KEY=your_google_ai_studio_key

## 📁 Project Structure
ai-finance-assistant/
├── backend/
│   ├── main.py        # FastAPI app
│   ├── agents.py      # LangGraph multi-agent system
│   ├── retriever.py   # FAISS vector search
│   ├── ingest.py      # PDF ingestion pipeline
│   └── schemas.py     # Pydantic models
├── frontend/
│   └── src/App.jsx    # React UI
├── data/documents/    # RBI PDF documents
└── requirements.txt
