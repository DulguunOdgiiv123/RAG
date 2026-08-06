from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good enough for learning
sentence = "The cat sat on the mat."
embedding = model.encode(sentence)

print(type(embedding))    # numpy array
print(embedding.shape)    # (384,) — this model outputs 384-dim vectors
print(embedding[:10])     # look at actual numbers — just floats, nothing magic
