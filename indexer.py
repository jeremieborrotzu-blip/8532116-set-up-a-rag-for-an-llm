# indexer.py
import argparse
import logging
from typing import Optional

from utils.config import INPUT_DIR # INPUT_DATA_URL (uncomment if needed)
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_indexing(input_directory: str, data_url: Optional[str] = None):
    """Runs the full indexing process."""
    logging.info("--- Starting the indexing process ---")

    # --- Step 1: Download and Extract (Optional) ---
    if data_url:
        logging.info(f"Attempting to download from the URL: {data_url}")
        success = download_and_extract_zip(data_url, input_directory)
        if not success:
            logging.error("Download or extraction failed. Stopping.")
            # Decide whether to continue with the existing local content or to stop.
            # Here we stop to avoid indexing potentially incomplete/old data.
            return
    else:
        logging.info(f"No URL provided. Using the local files in: {input_directory}")

    # --- Step 2: Loading and Parsing the Files ---
    logging.info(f"Loading and parsing the files from: {input_directory}")
    documents = load_and_parse_files(input_directory)

    if not documents:
        logging.warning("No document was loaded or parsed. Check the content of the input folder.")
        logging.info("--- Indexing process finished (no document processed) ---")
        return

    # --- Step 3: Creating/Updating the Vector index ---
    logging.info("Initializing the Vector Store manager...")
    vector_store = VectorStoreManager() # The constructor only loads if it exists

    logging.info("Building the Faiss index (this may take some time)...")
    # This method will split, generate the embeddings, create the index and save
    vector_store.build_index(documents)

    logging.info("--- Indexing process finished successfully ---")
    logging.info(f"Number of documents processed: {len(documents)}")
    if vector_store.index:
        logging.info(f"Number of indexed chunks: {vector_store.index.ntotal}")
    else:
        logging.warning("The final index could not be created or is empty.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexing script for the RAG application")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Directory containing the source files (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        # default=INPUT_DATA_URL, # Uncomment to use the value from the .env by default
        default=None,
        help="Optional URL to download and extract an inputs.zip file"
    )
    args = parser.parse_args()

    # Check whether the URL is passed as an argument, otherwise take the one from .env (if set)
    # final_data_url = args.data_url if args.data_url is not None else INPUT_DATA_URL
    # Simplification: we only use the --data-url argument for now
    final_data_url = args.data_url

    run_indexing(input_directory=args.input_dir, data_url=final_data_url)
