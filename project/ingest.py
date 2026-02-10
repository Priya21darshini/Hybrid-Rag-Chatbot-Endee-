from sentence_transformers import SentenceTransformer
import numpy as np
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_STORE_PATH = "project/vector_store.npy"
DOCS_PATH = "project/data/docs"

def chunk_text(text, chunk_size=40):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

vector_store = []

for file in os.listdir(DOCS_PATH):
    with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    for chunk in chunks:
        embedding = model.encode(chunk)
        vector_store.append({
            "source": file,
            "content": chunk,
            "embedding": embedding
        })

np.save(VECTOR_STORE_PATH, vector_store)

print("✅ Chunked documents stored successfully.")
print(f"Total chunks stored: {len(vector_store)}")
