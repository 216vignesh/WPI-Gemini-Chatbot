import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
import uuid

# ------------------------------
# Configuration
# ------------------------------

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-east1-aws"
INDEX_NAME = "events-index"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

URLS = [
    "https://www.wpi.edu/offices/international-student-scholars-office",
    "https://www.wpi.edu/offices/international-student-scholars-office#connect",
    "https://www.wpi.edu/offices/international-house/student/forms",
    "https://www.aclu.org/know-your-rights/immigrants-rights",
    "https://www.aclu.org/know-your-rights/what-do-when-encountering-law-enforcement-airports-and-other-ports-entry-us",
    "https://www.aclunc.org/our-work/know-your-rights/know-your-rights-us-airports-and-ports-entry",
    "https://www.aclu.org/know-your-rights/stopped-by-police",
    "https://www.aclum.org/en/know-your-rights/know-your-rights-if-you-are-questioned-about-your-immigration-status",
    "https://www.aclu.org/documents/constitution-100-mile-border-zone",
    "https://www.aclum.org/en/know-your-rights",
    "https://www.mass.gov/info-details/tenant-rights",
    "https://www.mass.gov/info-details/massachusetts-identification-id-requirements",
    "https://www.wpi.edu/sites/default/files/inline-image/Student-Experiences/International-House/2019.7%20Getting%20a%20MA%20ID%20and%20Liquor%20ID.pdf",
    "https://www.uscis.gov/ar-11",
    "https://www.wpi.edu/offices/international-student-scholars-office#mail",
    "https://www.wpi.edu/offices/international-student-scholars-office#payment",
    "https://www.wpi.edu/offices/international-house/faculty/immigration-overview",
    "https://www.wpi.edu/offices/international-house/students/immigration/non-immigrant-visa",
    "https://www.wpi.edu/offices/international-house/students/immigration/student-exchange-visitor-program-sevis",
    "https://www.wpi.edu/offices/international-house/students/immigration/f1-status",
    "https://www.wpi.edu/offices/international-house/students/immigration/j1-status",
    "https://www.wpi.edu/offices/international-house/students/immigration/social-security",
    "https://www.wpi.edu/offices/international-house/students/immigration/traveling",
    "https://www.wpi.edu/offices/international-house/students/getting-a-visa",
    "https://www.wpi.edu/offices/international-house/students/traveling",
    "https://www.wpi.edu/offices/international-house/students/traveling/graduate-international-orientation",
    "https://www.wpi.edu/offices/international-house/faculty/health-insurance-care-us",
    "https://www.wpi.edu/offices/international-house/faculty/worcester-community-info",
    "https://www.wpi.edu/offices/international-house/faculty/useful-links",
    "https://www.wpi.edu/offices/international-house/meet-the-team",
    "https://www.wpi.edu/offices/international-house/faculty/forms"
]

# ------------------------------
# Initialize Pinecone and Embedder
# ------------------------------

pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ------------------------------
# Remove Last Entered Data
# ------------------------------

def remove_last_entries():
    # Retrieve all vectors (limited to 10,000 for demonstration)
    existing_ids = index.describe_index_stats().get("namespaces", {}).get("", {}).get("vector_count", 0)
    if existing_ids > 0:
        print(f"Deleting all existing vectors...")
        index.delete(delete_all=True)
        print("All vectors deleted successfully.")
    else:
        print("No vectors to delete.")

# ------------------------------
# Process a List of URLs
# ------------------------------

def process_urls(urls):
    data = []
    for url in tqdm(urls, desc="Processing URLs"):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract main content from the page
        main_content = soup.find("main") or soup
        text = " ".join(main_content.stripped_strings)
        data.append({"url": url, "text": text})

    return data

# ------------------------------
# Process Data and Append to Pinecone
# ------------------------------

def process_and_store_data(data):
    MAX_METADATA_SIZE = 40960  # Pinecone metadata size limit in bytes
    MAX_CHUNK_SIZE = 500  # Maximum characters per chunk

    documents = []
    for entry in data:
        text = entry["text"]
        url = entry["url"]

        # Split text into chunks if it exceeds the size limit
        start = 0
        while start < len(text):
            chunk_text = text[start:start + MAX_CHUNK_SIZE]
            doc_id = str(uuid.uuid4())

            # Ensure metadata fits within the size limit
            metadata = {
                "url": url,
                "source": "WPI ISSO Website",
                "Description": chunk_text[:MAX_METADATA_SIZE]  # Truncate to fit size limit
            }

            documents.append({
                "id": doc_id,
                "text": chunk_text,
                "metadata": metadata
            })

            start += MAX_CHUNK_SIZE

    # Generate embeddings
    corpus = [doc["text"] for doc in documents]
    embeddings = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)

    # Prepare and upload to Pinecone
    vectors = []
    for doc, emb in zip(documents, embeddings):
        vectors.append({
            "id": doc["id"],
            "values": emb.tolist(),
            "metadata": doc["metadata"]
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // batch_size + 1}")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    # print("Removing previous data...")
    # remove_last_entries()

    print("Processing URLs...")
    crawled_data = process_urls(URLS)
    print(f"Processed {len(crawled_data)} URLs.")

    print("Processing and storing data...")
    process_and_store_data(crawled_data)
    print("Data appended to Pinecone index successfully.")

if __name__ == "__main__":
    main()


