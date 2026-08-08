import chromadb
client = chromadb.Client()

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
collection =client.create_collection(name="my_documents")
collection.add(documents=documents,
               ids=[str(i) for i in range(len(documents))]
               )

print(collection.count())

result = collection.query(
    query_texts = ["What food helps you sleep better ?"],
    n_results = 3
)

print(result)
