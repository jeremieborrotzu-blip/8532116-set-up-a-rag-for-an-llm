# MistralChat.py (RAG version)
import streamlit as st
import os
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

# --- Imports from your modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, COMMUNE_NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Import error: {e}. Check your folder structure and the files in 'utils'.")
    st.stop()


# --- Logging Configuration ---
# Note: Streamlit may have its own log handling. Configuring here is good practice.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Mistral API Configuration ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Error: Mistral API key not found (MISTRAL_API_KEY). Please set it in the .env file.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logging.info("Mistral client initialized.")
except Exception as e:
    st.error(f"Error while initializing the Mistral client: {e}")
    logging.exception("Mistral client initialization error")
    st.stop()

# --- Loading the Vector Store (cached) ---
@st.cache_resource # Keeps the manager loaded in memory for the session
def get_vector_store_manager():
    logging.info("Attempting to load the VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Check whether the index was properly loaded by the constructor
        if manager.index is None or not manager.document_chunks:
            st.error("The vector index or the chunks could not be loaded.")
            st.warning("Make sure you have run 'python indexer.py' after placing your files in the 'inputs' folder.")
            logging.error("Faiss index or chunks not found/loaded by VectorStoreManager.")
            return None # Return None on failure
        logging.info(f"VectorStoreManager loaded successfully ({manager.index.ntotal} vectors).")
        return manager
    except FileNotFoundError:
         st.error("Index or chunks files not found.")
         st.warning("Please run 'python indexer.py' to create the knowledge base.")
         logging.error("FileNotFoundError during the VectorStoreManager init.")
         return None
    except Exception as e:
        st.error(f"Unexpected error while loading the VectorStoreManager: {e}")
        logging.exception("VectorStoreManager loading error")
        return None

vector_store_manager = get_vector_store_manager()

# --- System Prompt for RAG ---
# Adapt this prompt to your needs
SYSTEM_PROMPT = f"""You are an expert virtual assistant for {COMMUNE_NAME} City Hall.
Your mission is to answer residents' questions accurately, factually and politely, based **exclusively** on the information provided in the CONTEXT below.

Important instructions:
1.  **Base your answer ONLY on the CONTEXT.** Do not make up any information.
2.  If the CONTEXT contains the information to answer the QUESTION, summarize it clearly.
3.  If the CONTEXT does NOT contain relevant information to answer the QUESTION, politely reply that you did not find the information in the current knowledge base and suggest contacting the City Hall services directly. Do not look for the answer elsewhere.
4.  Do not answer off-topic questions (unrelated to City Hall or to the context information).
5.  If possible, mention the source (for example, the file name) if it is indicated in the context.
6.  Keep your answers concise and easy to understand.

PROVIDED CONTEXT:
---
{{context_str}}
---

RESIDENT'S QUESTION:
{{question}}

CITY HALL ASSISTANT'S ANSWER:"""


# --- Initializing the conversation history ---
if "messages" not in st.session_state:
    # Initial welcome message
    st.session_state.messages = [{"role": "assistant", "content": f"Hello, I am the virtual assistant for {COMMUNE_NAME} City Hall. How can I help you regarding our services (based on my knowledge base)?"}]

# --- Functions ---

def generate_response(prompt_messages: list[ChatMessage]) -> str:
    """
    Sends the prompt (which now includes the context) to the Mistral API.
    """
    if not prompt_messages:
         logging.warning("Attempt to generate an answer with an empty prompt.")
         return "I cannot process an empty request."
    try:
        logging.info(f"Calling the Mistral API model '{model}' with {len(prompt_messages)} message(s).")
        # Log the prompt content (may be long) - comment out if too verbose
        # logging.debug(f"Prompt sent to the API: {prompt_messages}")

        response = client.chat(
            model=model,
            messages=prompt_messages,
            temperature=0.1, # Low temperature for factual answers based on the context
            # top_p=0.9,
        )
        if response.choices and len(response.choices) > 0:
            logging.info("Answer received from the Mistral API.")
            return response.choices[0].message.content
        else:
            logging.warning("The API did not return a valid choice.")
            return "Sorry, I could not generate a valid answer at the moment."
    except Exception as e:
        st.error(f"Error while calling the Mistral API: {e}")
        logging.exception("Mistral API error during client.chat")
        return "I am sorry, a technical error prevents me from answering. Please try again later."

# --- Streamlit User Interface ---
st.title(APP_TITLE)
st.caption(f"Virtual assistant for {COMMUNE_NAME} | Model: {model}")

# Display the history messages (for the UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input area
if prompt := st.chat_input(f"Ask your question about {COMMUNE_NAME}..."):
    # 1. Add and display the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Start of the RAG logic ===

    # 2. Check whether the Vector Store is available
    if vector_store_manager is None:
        st.error("The knowledge search service is not available. Cannot process your request.")
        logging.error("VectorStoreManager not available for the search.")
        # We stop here because we cannot do RAG
        st.stop()

    # 3. Search for the context in the Vector Store
    try:
        logging.info(f"Searching context for the question: '{prompt}' with k={SEARCH_K}")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info(f"{len(search_results)} chunks found in the Vector Store.")
    except Exception as e:
        st.error(f"An error occurred while searching for relevant information: {e}")
        logging.exception(f"Error during vector_store_manager.search for the query: {prompt}")
        search_results = [] # We continue without context if the search fails

    # 4. Format the context for the LLM prompt
    context_str = "\n\n---\n\n".join([
        f"Source: {res['metadata'].get('source', 'Unknown')} (Score: {res['score']:.1f}%)\nContent: {res['text']}"
        for res in search_results
    ])

    if not search_results:
        context_str = "No relevant information found in the knowledge base for this question."
        logging.warning(f"No context found for the query: {prompt}")

    # 5. Build the final prompt for the Mistral API using the RAG System Prompt
    final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)

    # Create the list of messages for the API (just the combined system/user prompt)
    messages_for_api = [
        # We could separate system and user, but Mistral handles a long structured user message well
        ChatMessage(role="user", content=final_prompt_for_llm)
    ]

    # === End of the RAG logic ===


    # 6. Display indicator + Generate the assistant's answer via the LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Simple indicator

        # Generate the assistant's answer using the augmented prompt
        response_content = generate_response(messages_for_api)

        # Display the full answer
        message_placeholder.write(response_content)

    # 7. Add the assistant's answer to the history (for the UI display)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Small optional footer
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | " + COMMUNE_NAME + " City Hall")
