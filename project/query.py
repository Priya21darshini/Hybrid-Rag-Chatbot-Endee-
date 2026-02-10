from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = np.load("project/vector_store.npy", allow_pickle=True)

query = input("Enter your question: ")
query_embedding = model.encode(query)

results = sorted(
    vector_store,
    key=lambda x: np.dot(x["embedding"], query_embedding),
    reverse=True
)[:2]

print("\n🔍 Search Results:\n")
for r in results:
    print(r["content"])
