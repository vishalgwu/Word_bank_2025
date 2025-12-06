import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    return model.encode(text).tolist()


def retrieve_relevant(qdrant, collection_name: str, query: str, top_k: int = 5):
    query_vec = embed_text(query)

    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )

    docs = []
    for r in results:
        docs.append({
            "score": r.score,
            "text": r.payload.get("text", "")
        })

    return docs
