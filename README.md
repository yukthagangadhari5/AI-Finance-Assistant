# 🚀 AI Finance Assistant (Multi-Agent RAG System)

## 📌 Overview
An advanced AI-powered finance assistant built using Retrieval-Augmented Generation (RAG) and a multi-agent architecture. The system processes financial data and intelligently responds to user queries.

---

## ⚙️ Features
- 🧠 Multi-Agent System (Finance, Summary, Alert)
- 🔍 RAG-based Retrieval using FAISS
- ⚡ FastAPI Backend
- 📊 Financial Insights & Risk Detection

---

## 🧠 Agents

### 🟢 Finance Agent
Finds unpaid invoices

### 🔵 Summary Agent
Provides total invoices and amount

### 🔴 Alert Agent
Detects high unpaid risk

---

## 🛠️ Tech Stack
- Python
- FastAPI
- LangChain
- FAISS
- HuggingFace Embeddings

---

## 🧪 Example Queries
- Which invoices are unpaid?
- Give me summary
- Is there any risk?

---

## ▶️ Run Project

```bash
pip install fastapi uvicorn pandas langchain langchain-community sentence-transformers
uvicorn main:app --reload