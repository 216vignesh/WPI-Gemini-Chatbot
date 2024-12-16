# search_pinecone.py

import os
import json
# import pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone
# ------------------------------
# Configuration
# ------------------------------

# Load environment variables from .env
load_dotenv()

# Pinecone API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east1-aws"

# Pinecone Index Configuration
INDEX_NAME = "events-index"

# Embedding model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# ------------------------------
# Initialize Pinecone
# ------------------------------

if not PINECONE_API_KEY:
    raise ValueError("Please set the PINECONE_API_KEY in your .env file.")
if not PINECONE_ENV:
    raise ValueError("Please set the PINECONE_ENV in your .env file.")
pc=Pinecone(api_key=PINECONE_API_KEY)
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Connect to the index
# if INDEX_NAME not in pc.list_indexes():
#     raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist.")
index = pc.Index(INDEX_NAME)

# ------------------------------
# Step 1: Define Semantic Search Function
# ------------------------------

def semantic_search(query, embedder, index, top_k=5):
    # Generate embedding for the query
    query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()
    
    # Query Pinecone
    response = index.query(
        vector=query_embedding[0],
        top_k=top_k,
        include_metadata=True
    )
    
    results = []
    for match in response.matches:
        results.append({
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
            "text": match.metadata.get("Description", "")
        })
    return results

# ------------------------------
# Step 2: Example Usage
# ------------------------------

if __name__ == "__main__":
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    user_query = "Timing for IH"
    results = semantic_search(user_query, embedder, index, top_k=3)
    for res in results:
        print(f"Score: {res['score']}")
        print(f"Metadata: {res['metadata']}")
        print(f"Text snippet: {res['text']}\n---")
