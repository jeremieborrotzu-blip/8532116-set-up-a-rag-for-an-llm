# utils/config.py
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- API Key ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Warning: The Mistral API key (MISTRAL_API_KEY) is not set in the .env file")
    # You may choose to raise an exception here or continue with limited features
    # raise ValueError("Missing Mistral API key. Please set it in the .env file")

# --- Mistral Models ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest" # Or another model such as mistral-large-latest

# --- Indexing Configuration ---
# INPUT_DATA_URL = os.getenv("INPUT_DATA_URL") # Uncomment if you use a URL
INPUT_DIR = "inputs"                # Folder for the source data after extraction
VECTOR_DB_DIR = "vector_db"         # Folder to store the Faiss index and the chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                   # Chunk size in *characters* (aims for ~512 tokens)
CHUNK_OVERLAP = 150                 # Overlap in *characters*
EMBEDDING_BATCH_SIZE = 32           # Batch size for the embedding API

# --- Search Configuration ---
SEARCH_K = 5                        # Default number of documents to retrieve

# --- Database Configuration ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL for SQLAlchemy

# --- Application Configuration ---
APP_TITLE = "RAG Assistant"
COMMUNE_NAME = "Willow Creek" # Name to customize in the interface
