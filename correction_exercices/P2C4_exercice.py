# utils/vector_store.py
import numpy as np
import faiss
import logging
import pickle
from mistralai.client import MistralClient
# Make sure the embed_texts_mistral function is defined or imported here

# ... (other functions such as load_faiss_index, load_chunks, embed_texts_mistral) ...

def search_similar_documents(
	query: str,
	index: faiss.Index,
	chunks: list[str],
	client: MistralClient,
	k: int = 3,
	model_embed: str = "mistral-embed",
	max_distance_threshold: float | None = None # New parameter
	) -> list[str]:
	"""
	Searches the k most relevant chunks for a given query,
	with optional filtering by a max distance threshold.

	Args:
    	query: The user's question.
    	index: The loaded Faiss index.
    	chunks: The full list of text chunks.
    	client: The Mistral client for the embedding.
    	k: Maximum number of documents to return before filtering by threshold.
    	model_embed: Mistral embedding model to use.
    	max_distance_threshold: Max L2 distance threshold. If None, no filtering.

	Returns:
    	List of the filtered relevant chunks.
	"""
	if index is None:
    	logging.warning("Index not available for the search.")
    	return []
	try:
    	# Get the query embedding
    	query_embedding = embed_texts_mistral([query], client, model_embed)[0]
    	query_embedding_np = np.array([query_embedding]).astype('float32')

    	# Initial search for the k nearest neighbors
    	distances, indices = index.search(query_embedding_np, k)

    	# Extract the results of the first (and only) query
    	result_distances = distances[0]
    	result_indices = indices[0]

    	# Filtering by distance threshold
    	relevant_chunks = []
    	filtered_indices_distances = [] # For the logging/debug

    	for i in range(len(result_indices)):
        	idx = result_indices[i]
        	dist = result_distances[i]

        	# Ignore the invalid Faiss results (sometimes -1)
        	if idx == -1:
            	continue

        	# Apply the threshold filter if defined
        	if max_distance_threshold is None or dist <= max_distance_threshold:
            	relevant_chunks.append(chunks[idx])
                filtered_indices_distances.append((idx, dist))
        	else:
            	# If we reach a document that does not pass the threshold,
            	# we could potentially stop because the next ones are farther away.
            	# But Faiss does not always guarantee a perfect order for some indexes,
            	# so it is safer to check all k results.
            	logging.debug(f"Document index {idx} discarded (distance {dist:.4f} > threshold {max_distance_threshold})")


    	logging.info(f"RAG search: {len(relevant_chunks)}/{k} chunks found after filtering (threshold={max_distance_threshold}). Indices/Distances kept: {filtered_indices_distances}")
    	return relevant_chunks

	except Exception as e:
    	logging.error(f"Error during the similarity search: {e}")
    	return []
2. Call from MistralChat.py or the evaluation script
Make sure to read the values of k and the threshold from your configuration and pass them to the function.
Python
# Excerpt from MistralChat.py or generate_evaluation_data

# Load the parameters from the configuration
num_relevant_docs = config.get('rag_num_docs', 3) # k
distance_threshold = config.get('rag_distance_threshold', None) # Threshold (may be None)

# ... (after obtaining the 'prompt' or the 'question') ...

# --- Step 2: Retrieval (if RAG) ---
if intent == INTENT_RAG:
	logging.info(f"RAG intent: Searching documents (k={num_relevant_docs}, threshold={distance_threshold}).")
	model_embed = config.get('mistral_model_embed', 'mistral-embed')

	contexts = search_similar_documents(
    	prompt, # or 'q' in the evaluation function
    	index,
    	chunks,
    	client,
    	k=num_relevant_docs, # Pass k
    	model_embed=model_embed,
        max_distance_threshold=distance_threshold # Pass the threshold
	)
else:
	contexts = []
 # ... (the rest of the code stays identical: prompt building, generation) ...
