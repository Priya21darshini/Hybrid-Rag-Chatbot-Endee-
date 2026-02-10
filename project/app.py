import streamlit as st
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

VECTOR_STORE_PATH = "project/vector_store.npy"
THRESHOLD = 0.25
TOP_K = 2

# Load model and vector store
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    return np.load(VECTOR_STORE_PATH, allow_pickle=True)

model = load_model()
vector_store = load_vector_store()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_with_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# ---------------- UI ----------------

st.set_page_config(page_title="Hybrid RAG Chatbot")
st.title("🤖 Hybrid RAG Chatbot")
st.write("Document-grounded answers with intelligent fallback.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask something..."):
    
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    query_embedding = model.encode(prompt)

    scored_results = []
    for item in vector_store:
        score = cosine_similarity(item["embedding"], query_embedding)
        scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_results[:TOP_K]

    if top_results[0][0] >= THRESHOLD:
        # RAG mode
        context = " ".join([item["content"] for score, item in top_results])
        final_prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{prompt}
"""
        answer = generate_with_ollama(final_prompt)
    else:
        # General mode
        answer = generate_with_ollama(prompt)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
