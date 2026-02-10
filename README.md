#  Hybrid RAG Chatbot using Vector Search + Local LLM (Ollama)

##  Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) system** that combines:

- Semantic vector search
- Cosine similarity ranking
- Document chunking
- Threshold-based filtering
- Local Large Language Model (LLM) via Ollama
- Fallback general chat mode

The system retrieves relevant document chunks using vector similarity and generates grounded answers using a local LLM. If no relevant context is found, it switches to general conversational mode.

This architecture mimics how real-world AI assistants are built in production environments.

---

##  Problem Statement

Traditional chatbots often hallucinate or provide generic responses.  
This project solves that by:

- Retrieving relevant knowledge from a vector store
- Injecting retrieved context into the prompt
- Generating grounded, context-aware answers
- Falling back gracefully for out-of-domain queries

---

## Architecture

User Query  
→ Sentence Transformer Embedding  
→ Cosine Similarity Search (Vector Store)  
→ Threshold Check  

If Similarity ≥ Threshold  
→ Retrieve Top-K Relevant Chunks  
→ Combine Context  
→ Send Context + Query to Ollama (RAG Mode)  
→ Generate Grounded Response  

Else  
→ Send Query Directly to Ollama (General Mode)  
→ Generate General Response  

→ Final Response Returned to User



---

## 🛠 Tech Stack

### Core Technologies
- Python 3.x
- NumPy
- Requests

### NLP & Embeddings
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Hugging Face Transformers

### Vector Search
- Cosine Similarity (Custom Implementation)
- In-memory Vector Store (NumPy-based)

### Large Language Model
- Ollama
- Mistral (Local LLM)

### Frontend / Interface
- Streamlit (Chat-based UI)

### System Design
- Hybrid Retrieval-Augmented Generation (RAG)
- Threshold-based Relevance Filtering
- Top-K Retrieval Strategy

---


##  Key Concepts Implemented

### 1️ Document Chunking
Large documents are split into smaller semantic chunks to improve retrieval accuracy.

### 2️ Vector Embeddings
Each chunk is converted into a dense vector using Sentence Transformers.

### 3️ Cosine Similarity
Similarity between query embedding and stored embeddings is calculated using cosine similarity.

### 4️ Threshold Filtering
Only sufficiently relevant chunks are used for RAG to avoid hallucination.

### 5️ Hybrid Mode
If no relevant match is found:
- The system falls back to general LLM mode.
- Ensures smooth conversational behavior.

---

## Features

 Semantic search over documents  
 Chunk-based retrieval  
 Cosine similarity ranking  
 Threshold-based filtering  
 Hybrid RAG + General Chat  
 Fully offline (No paid APIs required)  
 Uses local LLM (Ollama)  
 Interactive CLI chatbot  

---

##  Project Structure

project/

├── data/docs/ # Knowledge base documents

├── ingest.py 

├── rag.py # Hybrid RAG chatbot logic

├── requirements.txt 


---

##  Setup Instructions

### Create Virtual Environment
python -m venv venv
.\venv\Scripts\Activate

### Install Dependencies
pip install -r project/requirements.txt

### Install Ollama (Local LLM)

Pull the model:
ollama pull mistral

Test the model:
ollama run mistral
(Type /exit to quit)

### Generate Vector Store
python project/ingest.py

### Run Hybrid RAG Chatbot
python project/rag.py

### Usage
- Ask AI/ML-related questions to trigger RAG mode (document-grounded answers).
- Ask general questions to trigger General Mode (LLM fallback).
- Type "exit" to close the chatbot.


##  Limitations

- The knowledge base is limited to the documents provided in `data/docs`
- Ollama requires local model execution and cannot be deployed on cloud-only platforms
- Response latency depends on local system resources

##  Future Improvements

- Add conversation memory across turns
- Support dynamic document upload
- Add evaluation metrics 
- Deploy using cloud-based LLM APIs
- Improve UI with advanced Streamlit components


### Demo Video
https://drive.google.com/drive/folders/1jU1toGVZiWerB2vfmVc8cm3cg1Mh_Ruh


