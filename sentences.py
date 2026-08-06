from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",       # paraphrase of #1 — should score high vs #1
    "I love programming in Python.",
    "Writing code in Python is fun.",     # paraphrase of #3 — should score high vs #3
    "The stock market crashed today.",    # unrelated to all
]

embeddings = model.encode(sentences)

def cosine_similarity(a, b):
    # write this yourself — dot product over product of norms
    #
    dot_product = np.dot(a,b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    
    cosine = dot_product / (magnitude_a * magnitude_b)
    return cosine



# compute and print pairwise similarity for every pair
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"[{i}] \"{sentences[i]}\"")
        print(f"[{j}] \"{sentences[j]}\"")
        print(f"similarity: {sim:.4f}\n")
