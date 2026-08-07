import numpy as np 
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
document_embedding = model.encode(documents)

def cosine_similarity(a,b):
    dot = np.dot(a,b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot/(magnitude_a*magnitude_b)

def search(query,documents,document_embedding,top_k = 3):
    query_embedding = model.encode(query)
    score = []
    for i in range(len(documents)):
        sim = cosine_similarity(query_embedding,document_embedding[i])
        score.append((documents[i],sim))
    score.sort(key =lambda x: x[1],reverse=True)
    return score[:top_k]

result = search("Here is query about running and health",documents,document_embedding,top_k=3)
for doc,sc in result:
    print(f"{sc:.4f} {doc}")
