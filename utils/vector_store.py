# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Used for the format expected by the splitter

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Manages the creation, loading and search of a Faiss index."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, any]] = []
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Loads the Faiss index and the chunks if the files exist."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Loading the Faiss index from {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Loading the chunks from {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vectors) and {len(self.document_chunks)} chunks loaded.")
            except Exception as e:
                logging.error(f"Error while loading the index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Faiss index or chunks files not found. The index is empty.")

    def _split_documents_to_chunks(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Splits the documents into chunks with metadata."""
        logging.info(f"Splitting {len(documents)} documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len, # Important: measure in characters
            add_start_index=True, # Adds the start position of the chunk in the original document
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            # Converts our document format to the Langchain Document format for the splitter
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc['metadata'].get('filename', 'N/A')}' split into {len(chunks)} chunks.")

            # Enriches each chunk with additional metadata
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": f"{doc_counter}_{i}", # Unique chunk identifier (doc_index_chunk_index)
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata, # Metadata inherited from the document (source, category, etc.)
                        "chunk_id_in_doc": i, # Position of the chunk within its original document
                        "start_index": chunk.metadata.get("start_index", -1) # Start position (in characters)
                    }
                }
                all_chunks.append(chunk_dict)
            doc_counter += 1

        logging.info(f"Total of {len(all_chunks)} chunks created.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """Generates the embeddings for a list of chunks via the Mistral API."""
        if not MISTRAL_API_KEY:
            logging.error("Cannot generate the embeddings: MISTRAL_API_KEY missing.")
            return None
        if not chunks:
            logging.warning("No chunk provided to generate the embeddings.")
            return None

        logging.info(f"Generating the embeddings for {len(chunks)} chunks (model: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Processing batch {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except MistralAPIException as e:
                logging.error(f"Mistral API error while generating embeddings (batch {batch_num}): {e}")
                logging.error(f"  Details: Status Code={e.status_code}, Message={e.message}")
            except Exception as e:
                logging.error(f"Unexpected error while generating embeddings (batch {batch_num}): {e}")
                 # Handle the error: here we add zero vectors so as not to block
                num_failed = len(texts_to_embed)
                if all_embeddings: # If we already have embeddings, take the dimension of the first one
                    dim = len(all_embeddings[0])
                else: # Otherwise we cannot determine the dimension, skip this batch
                     logging.error("Cannot determine the embeddings dimension, skipping the batch.")
                     continue
                logging.warning(f"Adding {num_failed} zero vectors of dimension {dim} for the failed batch.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)

            except Exception as e:
                logging.error(f"Unexpected error while generating embeddings (batch {batch_num}): {e}")
                # Handle as above
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                     logging.error("Cannot determine the embeddings dimension, skipping the batch.")
                     continue
                logging.warning(f"Adding {num_failed} zero vectors of dimension {dim} for the failed batch.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)


        if not all_embeddings:
             logging.error("No embedding could be generated.")
             return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings generated successfully. Shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, any]]):
        """Builds the Faiss index from the documents."""
        if not documents:
            logging.warning("No document provided to build the index.")
            return

        # 1. Split into chunks
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error("Splitting produced no chunk. Cannot build the index.")
            return

        # 2. Generate the embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Embedding generation problem. The number of embeddings does not match the number of chunks.")
            # Clean up to avoid an inconsistent state
            self.document_chunks = []
            self.index = None
            # Remove the potentially corrupted files
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
            return


        # 3. Create the Faiss index optimized for cosine similarity
        dimension = embeddings.shape[1]
        logging.info(f"Creating the Faiss index optimized for cosine similarity with dimension {dimension}...")

        # Normalize the embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create an index for cosine similarity (IndexFlatIP = dot product)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Faiss index created with {self.index.ntotal} vectors.")

        # 4. Save the index and the chunks
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Saves the Faiss index and the list of chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Attempt to save an empty index or empty chunks.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Saving the Faiss index to {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Saving the chunks to {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index and chunks saved successfully.")
        except Exception as e:
            logging.error(f"Error while saving the index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        """
        Searches the k most relevant chunks for a query.

        Args:
            query_text: Query text
            k: Number of results to return
            min_score: Minimum score (between 0 and 1) to include a result

        Returns:
            List of relevant chunks with their scores
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Search impossible: the Faiss index is not loaded or is empty.")
            return []
        if not MISTRAL_API_KEY:
             logging.error("Search impossible: MISTRAL_API_KEY missing to generate the query embedding.")
             return []

        logging.info(f"Searching the {k} most relevant chunks for: '{query_text}'")
        try:
            # 1. Generate the query embedding
            response = self.mistral_client.embeddings(
                model=EMBEDDING_MODEL,
                input=[query_text] # The query must be a list
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            # Normalize the query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)

            # 2. Search in the Faiss index
            # For IndexFlatIP: scores = dot product (higher = better)
            # indices: indices of the matching chunks in self.document_chunks
            # Request more results if a minimum score is specified
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Format the results
            results = []
            if indices.size > 0: # Check whether there are results
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks): # Check the validity of the index
                        chunk = self.document_chunks[idx]
                        # Convert the score to a similarity (0-1)
                        # For IndexFlatIP with normalized vectors, the score is already between -1 and 1
                        # We convert it to a percentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filter the results based on the minimum score
                        # min_score is between 0 and 1, but similarity is a percentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(f"Document filtered out (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity, # Similarity score as a percentage
                            "raw_score": raw_score, # Raw score for debugging
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contains source, category, chunk_id_in_doc, start_index etc.
                        })
                    else:
                        logging.warning(f"Faiss index {idx} out of bounds (number of chunks: {len(self.document_chunks)}).")

            # Sort by score (highest similarity first)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limit to the requested number (k) if necessary
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} relevant chunks found (minimum score: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} relevant chunks found.")

            return results

        except MistralAPIException as e:
            logging.error(f"Mistral API error while generating the query embedding: {e}")
            logging.error(f"  Details: Status Code={e.status_code}, Message={e.message}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error during the search: {e}")
            return []
