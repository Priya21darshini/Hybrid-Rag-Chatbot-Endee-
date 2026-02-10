from sentence_transformers import SentenceTransformer
import numpy as np
import requests

VECTOR_STORE_PATH = "project/vector_store.npy"
THRESHOLD = 0.25
TOP_K = 2

model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = np.load(VECTOR_STORE_PATH, allow_pickle=True)

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

print("\n🚀 Hybrid RAG Chatbot Ready (Documents + General AI)\n")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()

    if query.lower() == "exit":
        print("Goodbye 👋")
        break

    if not query:
        print("\nPlease enter a valid question.\n")
        continue

    # Step 1: Embed query
    query_embedding = model.encode(query)

    # Step 2: Score against vector store
    scored_results = []
    for item in vector_store:
        score = cosine_similarity(item["embedding"], query_embedding)
        scored_results.append((score, item))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_results[:TOP_K]

    # Step 3: Decide mode
    if top_results[0][0] >= THRESHOLD:
        # ✅ RAG MODE (document-grounded)
        context = " ".join([item["content"] for score, item in top_results])

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""
        answer = generate_with_ollama(prompt)

        print("\n📄 Retrieved Context:\n")
        for score, item in top_results:
            print(f"[Score: {score:.3f}] {item['content']}\n")

        print("🤖 RAG Answer:\n")
        print(answer)

    else:
        # 🔄 GENERAL CHAT MODE (fallback)
        prompt = f"Answer this normally:\n{query}"
        answer = generate_with_ollama(prompt)

        print("\n🤖 General AI Answer:\n")
        print(answer)
