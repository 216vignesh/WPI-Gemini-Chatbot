# index_courses_pinecone.py

import requests
import json
import re
import uuid
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
import os
from dotenv import load_dotenv

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
VECTOR_DIMENSION = 384  # For 'all-MiniLM-L6-v2'
METRIC = "cosine"  # Options: "cosine", "dotproduct", "euclidean"

# URL to fetch data
DATA_URL = "https://courselistings.wpi.edu/assets/prod-data.json"

# Embedding model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Chunking configuration
MAX_CHARS = 500

# ------------------------------
# Validate Configuration
# ------------------------------

if not PINECONE_API_KEY:
    raise ValueError("Please set the PINECONE_API_KEY in your .env file.")
if not PINECONE_ENV:
    raise ValueError("Please set the PINECONE_ENV in your .env file.")

# ------------------------------
# Initialize Pinecone
# ------------------------------
pc=Pinecone(api_key=PINECONE_API_KEY)


# # Create index if it doesn't exist
# if INDEX_NAME not in pc.list_indexes():
#     pc.create_index(name=INDEX_NAME, dimension=VECTOR_DIMENSION, metric=METRIC)
#     print(f"Created Pinecone index '{INDEX_NAME}'.")
# else:
#     print(f"Pinecone index '{INDEX_NAME}' already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# ------------------------------
# Step 1: Fetch Data from Website
# ------------------------------

print("Fetching data from website...")
response = requests.get(DATA_URL)
response.raise_for_status()
data = response.json()

# Extract course entries
entries = data.get("Report_Entry", [])
print(f"Number of course entries fetched: {len(entries)}")

# ------------------------------
# Step 2: Clean and Preprocess Data
# ------------------------------

def clean_html(html_str):
    soup = BeautifulSoup(html_str, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

documents = []
for entry in tqdm(entries, desc="Processing course entries"):
    # Extract fields
    course_title = entry.get("Course_Title", "")
    course_desc_raw = entry.get("Course_Description", "")
    cleaned_desc = clean_html(course_desc_raw)
    
    instructors = entry.get("Instructors", "")
    locations = entry.get("Locations", "")
    meeting_patterns = entry.get("Meeting_Patterns", "")
    delivery_mode = entry.get("Delivery_Mode", "")
    academic_level = entry.get("Academic_Level", "")
    credits = entry.get("Credits", "")
    enrolled_capacity = entry.get("Enrolled_Capacity", "")
    section_status = entry.get("Section_Status", "")
    course_tags = entry.get("Course_Tags", "")
    academic_units = entry.get("Academic_Units", "")
    course_section = entry.get("Course_Section", "")
    offering_period = entry.get("Offering_Period", "")
    section_details = entry.get("Section_Details", "")
    start_date = entry.get("Course_Section_Start_Date", "")
    end_date = entry.get("Course_Section_End_Date", "")

    # Combine into a single text block
    full_text = (
        f"Title: {course_title}\n"
        f"Instructors: {instructors}\n"
        f"Locations: {locations}\n"
        f"Meeting Patterns: {meeting_patterns}\n"
        f"Delivery Mode: {delivery_mode}\n"
        f"Academic Level: {academic_level}\n"
        f"Credits: {credits}\n"
        f"Enrolled Capacity: {enrolled_capacity}\n"
        f"Section Status: {section_status}\n"
        f"Course Tags: {course_tags}\n"
        f"Academic Units: {academic_units}\n"
        f"Course Section: {course_section}\n"
        f"Offering Period: {offering_period}\n"
        f"Section Details: {section_details}\n"
        f"Start Date: {start_date}\n"
        f"End Date: {end_date}\n\n"
        f"Description:\n{cleaned_desc}"
    )

    metadata = {
        "course_title": course_title,
        "instructors": instructors,
        "locations": locations,
        "meeting_patterns": meeting_patterns,
        "offering_period": offering_period,
        "course_section": course_section,
        "academic_level": academic_level,
        "credits": credits,
        "section_status": section_status,
        "academic_units": academic_units,
        "course_tags": course_tags,
        "enrolled_capacity": enrolled_capacity,
        "delivery_mode": delivery_mode,
        "section_details": section_details,
        "start_date": start_date,
        "end_date": end_date
    }

    documents.append({
        "id": str(uuid.uuid4()),
        "text": full_text,
        "metadata": metadata
    })

print(f"Total documents after processing: {len(documents)}")

# ------------------------------
# Step 3: Chunk Data for Embedding
# ------------------------------

chunked_docs = []
for doc in tqdm(documents, desc="Chunking documents"):
    text = doc["text"]
    metadata = doc["metadata"]
    start = 0
    while start < len(text):
        end = start + MAX_CHARS
        chunk_text = text[start:end]
        start = end
        chunked_docs.append({
            "id": doc["id"] + f"_{start}",
            "text": chunk_text,
            "metadata": {**metadata, "Description": chunk_text}  # Include text in metadata
        })

print(f"Total chunks created: {len(chunked_docs)}")

# ------------------------------
# Step 4: Generate Embeddings
# ------------------------------

print("Generating embeddings...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
corpus = [d['text'] for d in tqdm(chunked_docs, desc="Preparing corpus")]

embeddings = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# ------------------------------
# Step 5: Upload to Pinecone
# ------------------------------

print("Uploading embeddings to Pinecone...")
vectors = []
for doc, emb in tqdm(zip(chunked_docs, embeddings), total=len(chunked_docs), desc="Preparing vectors"):
    vectors.append({
        "id": doc['id'],
        "values": emb.tolist(),
        "metadata": doc['metadata']
    })

# Pinecone supports upsert in batches
batch_size = 1000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted batch {i // batch_size + 1}")

print("All vectors uploaded to Pinecone successfully.")