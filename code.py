import os
from dotenv import load_dotenv
# 1. Load the environment keys FIRST
load_dotenv()

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# 2. Initialize the model
embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

# 3. Added missing commas at the end of each string item below
document = [
    "virat kholi is an indian cricketer known for his aggresive batting and leardership",
    "ms dhoni is a former indian captian famous for his calm demeanor and finishing abilities",
    "sachin tendulkar is a legendary indian batsman often referred to as the god of cricket",
    "rohit sharma is an indian cricketer known for his stylish batting and record-breaking performances",
    "jasprit bumrah is an indian fast bowler known for his deadly yorkers and pace"
]

query = 'tell me abpout virat kholi'

# 4. Generate the embeddings vectors
doc_embeddings = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)

# 5. Calculate structural similarity matrices
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# 6. Extract the highest matching profile
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("\n--- Most Similar Document ---")
print(document[index])
print("Similarity Score:", round(score, 4))
