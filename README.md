# Enterprise RAG AI Assistant

An end-to-end Retrieval-Augmented Generation (RAG) AI system built using LangChain, Groq LLMs, FAISS vector database, and Streamlit.

This project allows users to upload documents such as PDF or TXT files and ask questions in natural language. The system retrieves the most relevant document chunks using semantic search and generates contextual answers using a Large Language Model (LLM).

---

# Features

- Upload PDF and TXT documents
- Semantic document retrieval using FAISS
- AI-powered question answering
- Fast inference using Groq LLMs
- Source-aware retrieval pipeline
- Streamlit web interface
- Modular backend architecture
- Public deployment support

---

# Tech Stack

## AI / LLM

- Groq API
- Llama 3.1 8B Instant
- LangChain

## Vector Database

- FAISS (Facebook AI Similarity Search)

## Embeddings

- Google Gemini Embeddings
- `models/gemini-embedding-001`

## Frontend

- Streamlit

## Backend

- Python

---

# Project Architecture

```text
User Question
       │
       ▼
Document Retrieval (FAISS)
       │
       ▼
Relevant Chunks Retrieved
       │
       ▼
Prompt Construction
       │
       ▼
Groq LLM Generates Answer
       │
       ▼
Final Response Displayed