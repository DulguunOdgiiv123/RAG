"""
Semantic search over a small document set — implemented two ways:
1. From scratch with numpy (cosine similarity)
2. With FAISS (production vector search library)
Confirms both approaches return identical rankings.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    # cooking
    "Simmer the tomato sauce for twenty minutes before adding basil.",
    "A cast iron skillet retains heat better than nonstick pans.",
    "Kneading dough properly develops the gluten structure needed for bread.",
    # space
    "The James Webb telescope captured images of a distant galaxy cluster.",
    "Mars has two small moons named Phobos and Deimos.",
    "A neutron star forms when a massive star's core collapses.",
    # finance
    "The central bank raised interest rates to combat inflation.",
    "Diversifying a portfolio reduces exposure to single-stock risk.",
    "Quarterly earnings reports influence short-term stock price movements.",
    # sports
    "The striker scored a hat-trick in the second half.",
    "Marathon runners often hit a wall around the thirty-kilometer mark.",
    "The team's defense allowed the fewest goals in the league this season.",
    # health
    "Getting seven to nine hours of sleep supports immune function.",
    "Regular strength training helps preserve bone density with age.",
    "Chronic stress can elevate cortisol levels over time.",
]

doc_embeddings = model.encode(documents)


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot / (magnitude_a * magnitude_b)


def search_manual(query, documents, doc_embeddings, top_k=3):
    """Brute-force semantic search, cosine similarity computed by hand."""
    query_embedding = model.encode(query)
    scores = []
    for i in range(len(documents)):
        sim = cosine_similarity(query_embedding, doc_embeddings[i])
        scores.append((documents[i], sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_faiss_index(doc_embeddings):
    """Normalize embeddings and build a FAISS flat inner-product index.
    Normalized dot product == cosine similarity."""
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)
    return index


def search_faiss(query, documents, index, top_k=3):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [(documents[idx], score) for idx, score in zip(indices[0], distances[0])]


if __name__ == "__main__":
    query = "What foods help you sleep better?"

    print("--- Manual (numpy) ---")
    for doc, score in search_manual(query, documents, doc_embeddings, top_k=3):
        print(f"{score:.4f}  {doc}")

    index = build_faiss_index(doc_embeddings)

    print("\n--- FAISS ---")
    for doc, score in search_faiss(query, documents, index, top_k=3):
        print(f"{score:.4f}  {doc}")
