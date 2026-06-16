# app.py
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import logging
import datetime
from streamlit_feedback import streamlit_feedback # Import the component

# Import our local modules
from utils.config import APP_TITLE, COMMUNE_NAME, MISTRAL_API_KEY
from utils.vector_store import VectorStoreManager
from utils.database import log_interaction, update_feedback # Import update_feedback
from utils.query_classifier import QueryClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit page configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide"
)

# --- Initialization (with Streamlit caching) ---

# Caches the VectorStoreManager to avoid reloading the index on every interaction
@st.cache_resource
def get_vector_store():
    logging.info("Loading the VectorStoreManager...")
    return VectorStoreManager()

# Caches the Mistral client
@st.cache_resource
def get_mistral_client():
    if not MISTRAL_API_KEY:
        st.error("Error: The Mistral API key (MISTRAL_API_KEY) is not configured.")
        st.stop()
    logging.info("Initializing the Mistral client...")
    return MistralClient(api_key=MISTRAL_API_KEY)

# Caches the query classifier
@st.cache_resource
def get_query_classifier():
    logging.info("Initializing the query classifier...")
    return QueryClassifier()

# Loads the Vector Store, the Mistral client and the query classifier
vector_store = get_vector_store()
client = get_mistral_client()
query_classifier = get_query_classifier()

# Initializes the chat history in the session state if it does not exist
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initializes the ID of the last interaction for the feedback
if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None

# --- User Interface ---

# Sidebar
with st.sidebar:
    st.title(f"📚 {COMMUNE_NAME}")
    st.caption(f"Municipal virtual assistant")

    # Button to start a new conversation
    if st.button("🔄 New conversation", use_container_width=True):
        # Reset the message history
        st.session_state.messages = []
        st.session_state.last_interaction_id = None
        st.rerun()  # Reload the app to display the new conversation

    st.divider()

    # Application settings
    st.subheader("⚙️ Settings")

    # Mistral model selector
    model_options = {
        "mistral-small-latest": "Mistral Small (fast)",
        "mistral-large-latest": "Mistral Large (accurate)"
    }
    selected_model = st.selectbox(
        "LLM model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0  # Small by default
    )

    # Slider for the number of documents
    num_docs = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=20,
        value=5,  # 5 by default
        step=1
    )

    # Slider for the minimum score (as a percentage)
    min_score_percent = st.slider(
        "Minimum score (filter out weak results)",
        min_value=0,
        max_value=100,
        value=75,  # 75% by default
        step=5,
        format="%d%%"
    )
    # Convert the percentage to a decimal value (0-1)
    min_score = min_score_percent / 100.0

    st.divider()

    # Information about the application
    st.subheader("📝 Information")
    st.markdown(f"**Selected model**: {model_options[selected_model]}")
    st.markdown(f"**Indexed documents**: {vector_store.index.ntotal if vector_store.index else 0}")

    # Information about the current conversation
    if st.session_state.messages:
        st.info(f"{len(st.session_state.messages) // 2} exchanges in this conversation")

        # Button to download the conversation
        # Prepare the conversation content in text format
        conversation_text = "\n\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.messages
        ])

        # Add a header with the date and the title
        header = f"Conversation with the virtual assistant of {COMMUNE_NAME}\n"
        header += f"Date: {datetime.datetime.now().strftime('%m/%d/%Y %H:%M')}\n\n"
        conversation_text = header + conversation_text

        # Download button
        st.download_button(
            label="💾 Download the conversation",
            data=conversation_text,
            file_name=f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Main title
st.title(f"📚 {APP_TITLE}")
st.caption(f"Ask your questions about {COMMUNE_NAME}")

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display the sources if they exist for the assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Sources used"):
                for i, source in enumerate(message["sources"]):
                    # Safe access to the metadata
                    meta = source.get("metadata", {})
                    st.markdown(f"**Source {i+1}:** `{meta.get('source', 'N/A')}`")
                    st.markdown(f"*Similarity score:* {source.get('score', 0.0):.2f}%")
                    if 'raw_score' in source:
                        st.markdown(f"*Raw score:* {source.get('raw_score', 0.0):.4f}")
                    st.markdown(f"*Category:* `{meta.get('category', 'N/A')}`")
                    st.text_area(f"Excerpt {i+1}", value=source.get("text", "")[:500]+"...", height=100, disabled=True, key=f"src_{message['timestamp']}_{i}") # Unique key to avoid conflicts


# User input area at the bottom
if prompt := st.chat_input("Ask your question here..."):
    # Add the user message to the history and display it
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now().isoformat()})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a waiting message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Searching for information and generating the answer...")

        # --- Query processing logic ---
        try:
            # 1. Classify the query to determine whether it requires RAG
            needs_rag, confidence, reason = query_classifier.needs_rag(prompt)

            # Display the classification result
            mode_str = "RAG" if needs_rag else "DIRECT"
            logging.info(f"Query classification: {mode_str} (confidence: {confidence:.2f}) - Reason: {reason}")

            # Display a message indicating the mode used
            mode_info = st.empty()
            if needs_rag:
                mode_info.info(f"RAG mode: Searching for specific information in the knowledge base (confidence: {confidence:.2f})")
                # 2. Search in the Vector Store if necessary
                logging.info(f"Searching documents for: '{prompt}' (max: {num_docs}, min score: {min_score})")
                retrieved_docs = vector_store.search(prompt, k=num_docs, min_score=min_score)
            else:
                mode_info.info(f"Direct mode: Answer based on the model's general knowledge (confidence: {confidence:.2f})")
                # No search in the Vector Store
                retrieved_docs = []

            # 2. Prepare the data depending on the mode
            if needs_rag and retrieved_docs:
                # RAG mode with documents found
                logging.info(f"{len(retrieved_docs)} documents retrieved.")
                # Prepare the context for the LLM
                context_str = "\n\n---\n\n".join([
                    f"Source: {doc['metadata'].get('source', 'Unknown')} (Score: {doc['score']:.4f})\nContent: {doc['text']}"
                    for doc in retrieved_docs
                ])
                sources_for_log = [ # Simplified version for the log and the display
                    {"text": doc["text"], "metadata": doc["metadata"], "score": doc["score"]}
                    for doc in retrieved_docs
                ]

                # System prompt for RAG mode
                system_prompt = f"""You are a virtual assistant for {COMMUNE_NAME}.
Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say that you do not know or that the information is not available in the provided documents.
Be concise and accurate. Cite your sources if possible (for example, by mentioning the file name or the category found in the metadata).

Provided context:
---
{context_str}
---
"""
            elif needs_rag and not retrieved_docs:
                # RAG mode but no document found
                logging.warning("No relevant document found.")
                context_str = "No relevant information found in the documents."
                sources_for_log = []

                # System prompt for RAG mode with no results
                system_prompt = f"""You are a virtual assistant for {COMMUNE_NAME}.
The user asked a question that seems to concern information specific to the town, but no relevant information was found in our knowledge base.
Politely indicate that you do not have this specific information and suggest that the user rephrase their question or contact City Hall directly.
Do not make up information about {COMMUNE_NAME}.
"""
            else:
                # Direct mode (no RAG)
                context_str = "Direct mode: answer based on the model's general knowledge."
                sources_for_log = []

                # System prompt for Direct mode
                system_prompt = f"""You are a virtual assistant for {COMMUNE_NAME}.
Answer the user's question using your general knowledge.
Be concise, accurate and helpful.
If the question concerns information specific to {COMMUNE_NAME} that you do not know, clearly indicate that you do not have this specific information.
Do not make up information about {COMMUNE_NAME}.
"""
            user_message = ChatMessage(role="user", content=prompt)
            system_message = ChatMessage(role="system", content=system_prompt)
            messages_for_api = [system_message, user_message]

            # 3. Call the Mistral Chat API
            logging.info(f"Calling the Mistral Chat API with the model {selected_model}...")
            chat_response = client.chat(
                model=selected_model,
                messages=messages_for_api
            )
            response_text = chat_response.choices[0].message.content
            logging.info("Answer generated by Mistral.")

            # 4. Display the answer and the sources
            message_placeholder.markdown(response_text)

            # Display the sources if available (RAG mode with results)
            if sources_for_log:
                with st.expander("Sources used"):
                    for i, source in enumerate(sources_for_log):
                        meta = source.get("metadata", {})
                        st.markdown(f"**Source {i+1}:** `{meta.get('source', 'N/A')}`")
                        st.markdown(f"*Similarity score:* {source.get('score', 0.0):.2f}%")
                        if 'raw_score' in source:
                            st.markdown(f"*Raw score:* {source.get('raw_score', 0.0):.4f}")
                        st.markdown(f"*Category:* `{meta.get('category', 'N/A')}`")
                        st.text_area(f"Excerpt {i+1}", value=source.get("text", "")[:500]+"...", height=100, disabled=True, key=f"src_new_{i}") # Unique key
            elif needs_rag:
                # RAG mode with no results
                st.info("No relevant source was found in the knowledge base for this question.")
            else:
                # Direct mode
                st.info("Answer generated in direct mode, without consulting the knowledge base.")

            # 5. Log the interaction in the database (without initial feedback)
            # Add metadata about the mode used
            metadata = {
                "mode": "RAG" if needs_rag else "DIRECT",
                "confidence": confidence,
                "reason": reason
            }

            interaction_id = log_interaction(
                query=prompt,
                response=response_text,
                sources=sources_for_log, # Stores the list of dicts
                metadata=metadata # Add the metadata about the mode
            )
            st.session_state.last_interaction_id = interaction_id # Keep the ID for the feedback
            logging.info(f"Interaction logged with ID: {interaction_id}")


            # Add the assistant's answer to the history for permanent display
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources_for_log, # Keep the sources for redisplay
                "timestamp": datetime.datetime.now().isoformat(),
                 "interaction_id": interaction_id # Link the message to the DB ID
            })


        except Exception as e:
            # Check whether it is a Mistral API error
            if hasattr(e, 'status_code') and hasattr(e, 'message'):
                logging.error(f"Mistral API error: {e}")
                message_placeholder.error(f"An error occurred while communicating with the Mistral API: {e}")
            else:
                logging.error(f"Unexpected error: {e}", exc_info=True)
                message_placeholder.error(f"An error occurred: {e}")

            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "sources": [], "timestamp": datetime.datetime.now().isoformat(), "interaction_id": None})
            st.session_state.last_interaction_id = None # No ID if error before logging

# --- Feedback Section ---
# Place the feedback after the display loop and the chat input area
# We target the *last* assistant answer for the feedback
last_assistant_message = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)

# Check whether the last answer has an associated interaction ID
current_interaction_id = last_assistant_message.get("interaction_id") if last_assistant_message else None

if current_interaction_id:
    # Use streamlit-feedback
    feedback = streamlit_feedback(
        feedback_type="thumbs", # "thumbs" or "faces"
        optional_text_label="[Optional] Comments:",
        key=f"feedback_{current_interaction_id}", # Unique key linked to the interaction
        align="flex-start",  # Align to the left
        on_submit=lambda x: logging.info(f"Feedback submitted: {x}")  # Log for debugging
    )

    # Process the feedback if it is given
    if feedback:
        # Convert the feedback to a numeric value and text
        feedback_score = feedback.get('score')

        # Check whether the score is valid
        # The streamlit_feedback component may return emojis instead of "thumbs_up"/"thumbs_down"
        if feedback_score == "👍" or feedback_score == "thumbs_up":
            feedback_score = "positive"
        elif feedback_score == "👎" or feedback_score == "thumbs_down":
            feedback_score = "negative"
        else:
            logging.warning(f"Invalid feedback score: {feedback_score}")
            feedback_score = None

        # 1 for positive, 0 for negative
        feedback_value = 1 if feedback_score == "positive" else 0 if feedback_score == "negative" else None

        # Text for the database ("positive" or "negative")
        feedback_text = "positive" if feedback_score == "positive" else "negative" if feedback_score == "negative" else "N/A"

        # Emoji for the display in the interface
        feedback_emoji = "👍" if feedback_score == "positive" else "👎" if feedback_score == "negative" else "N/A"
        comment = feedback.get('text', None)

        # Update the interaction in the database
        success = update_feedback(current_interaction_id, feedback_text, comment, feedback_value)
        if success:
            st.toast(f"Thank you for your feedback ({feedback_emoji})!", icon="✅")
            # Optional: Disable the buttons after the first click to avoid multiple submissions
            # This is more complex to handle with the stateless nature of Streamlit without advanced callbacks.
            # For simplicity, we just log it. The user can re-click but only the last value counts.
        else:
            st.toast("Error while saving your feedback.", icon="❌")

        # Optional: Clear the feedback from the state to avoid resubmission on re-run
        # st.session_state[f"feedback_{current_interaction_id}"] = None # May cause issues if mishandled

else:
    st.write("Ask a question to be able to give your opinion on the answer.")
