# chatbot_pinecone.py

import os
import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# ------------------------------
# Configuration
# ------------------------------

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"

# Pinecone API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east1-aws"

# Pinecone Index Configuration
INDEX_NAME = "events-index"

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# ------------------------------
# Initialize Pinecone and Gemini
# ------------------------------

if not PINECONE_API_KEY:
    st.error("Please set the PINECONE_API_KEY in your .env file.")
    st.stop()
if not PINECONE_ENV:
    st.error("Please set the PINECONE_ENV in your .env file.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("Please set the GEMINI_API_KEY in your .env file.")
    st.stop()

pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# Initialize Embedder
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------------------
# Define Semantic Search Function
# ------------------------------

def semantic_search(query, embedder, index, top_k=10):
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
        # Handle cases where "Description" might be empty
        text = match.metadata.get("Description", "").strip()
        if not text:
            # Fallback to other metadata fields if "Description" is empty
            text = (
                f"Title: {match.metadata.get('course_title', 'Unknown')}\n"
                f"Section Details: {match.metadata.get('section_details', 'Unknown')}\n"
                f"Instructors: {match.metadata.get('instructors', 'Unknown')}\n"
            )
        results.append({
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
            "text": text
        })
    return results

# ------------------------------
# Define Function to Generate Response via Gemini
# ------------------------------

def generate_response_gemini(query, context):
    prompt = f"""Use the following context to answer the user's question. 
Please note:
- M in meeting patterns stands for Monday, 
- T for Tuesday, 
- W for Wednesday, 
- R for Thursday, 
- F for Friday.
Provide a detailed response for each question, also answer as naturally as possible as if a human is talking to a human. Do not use words like, from the given text, etc. Make it sound as natural as possible

Context:
{context}

User Query:
{query}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------------------
# Streamlit UI
# ------------------------------

def main():
    st.set_page_config(page_title="University Course Chatbot", layout="wide")
    st.title("🎓 University Course Chatbot")
    st.write("Ask me anything about university courses!")

    # User Input
    user_query = st.text_input("You:", "")

    if st.button("Send") and user_query:
        with st.spinner("Chatbot is thinking..."):
            # Perform semantic search
            search_results = semantic_search(
                query=user_query,
                embedder=embedder,
                index=index,
                top_k=5
            )

            if not search_results:
                st.write("Chatbot: I'm sorry, I couldn't find any relevant information.")
            else:
                # Compile context
                context = "\n\n".join([res['text'] for res in search_results if res['text']])
                
                if not context.strip():
                    st.write("Chatbot: I'm sorry, I couldn't find enough relevant context.")
                else:
                    # Generate response via Gemini
                    response = generate_response_gemini(
                        query=user_query,
                        context=context
                    )

                    # Display response
                    st.write(f"**Chatbot:** {response}")

if __name__ == "__main__":
    main()