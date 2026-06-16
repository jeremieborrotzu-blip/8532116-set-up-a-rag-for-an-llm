import streamlit as st
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import logging # Added for better debugging of API errors
from dotenv import load_dotenv
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)

# --- 1. Importing the libraries and configuration ---
st.set_page_config(page_title="City Hall Assistant", page_icon="🏛️")

# Retrieving the Mistral API key from the environment variables
# !! WARNING: Replace "YOUR_MISTRAL_API_KEY_HERE" with your key if you do not configure an environment variable !!
# It is STRONGLY recommended to use an environment variable.
api_key = os.environ.get("MISTRAL_API_KEY")

# Check for the presence of the API key
if not api_key:
    st.error("Mistral API key not found. Please set the MISTRAL_API_KEY environment variable.")
    # You can also offer a direct input (less secure)
    # api_key = st.text_input("Enter your Mistral API key:", type="password")
    # if not api_key:
    st.stop() # Stop the execution if the key is not provided

try:
    client = MistralClient(api_key=api_key)
    model = "mistral-large-latest" # Or another model such as "mistral-small-latest"
except Exception as e:
    st.error(f"Error while initializing the Mistral client: {e}")
    st.stop()

# --- 2. Initializing the conversation history ---
if "messages" not in st.session_state:
    # Adding an initial system message (optional but it can guide the model)
    # st.session_state.messages = [
    #     ChatMessage(role="system", content="You are a virtual assistant for City Hall. Answer residents' questions clearly and concisely.")
    # ]
    # Initialization with the assistant's welcome message
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am the City Hall virtual assistant. How can I help you today?"}]

# --- 3. Building the prompt with the history ---
def build_session_prompt(messages, max_messages=10):
    """
    Builds the prompt for the Mistral API using the recent messages.

    Args:
        messages (list): Full list of the session messages.
        max_messages (int): Maximum number of recent messages to include.

    Returns:
        list[ChatMessage]: List of messages formatted for the API.
    """
    # Keep only the last N messages to limit the prompt size
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    # Convert the dictionaries to ChatMessage objects
    formatted_messages = [
        ChatMessage(role=msg["role"], content=msg["content"])
        for msg in recent_messages
    ]

    # Optional: Add a system message at the beginning if not already done
    # if not any(m.role == "system" for m in formatted_messages):
    #     formatted_messages.insert(0, ChatMessage(role="system", content="You are a virtual assistant for City Hall. Answer residents' questions clearly and concisely."))

    logging.info(f"Messages sent to the API: {formatted_messages}") # For debugging
    return formatted_messages

# --- 4. Generating answers via the Mistral API ---
def generate_response(prompt_messages):
    """
    Calls the Mistral API to generate an answer.

    Args:
        prompt_messages (list[ChatMessage]): Formatted messages to send to the API.

    Returns:
        str: The content of the generated answer or an error message.
    """
    try:
        response = client.chat(
            model=model,
            messages=prompt_messages,
            # safe_prompt=True # Uncomment if you want to enable the safe mode
        )
        # Check whether the response contains choices
        if response.choices:
            return response.choices[0].message.content
        else:
            logging.error("The Mistral API returned no choice.")
            return "I am sorry, I could not generate an answer. No option returned."
    except Exception as e:
        logging.error(f"Error while calling the Mistral API: {e}")
        # Provide more details if possible, for example about quota errors
        st.error(f"Error while generating the answer: {e}")
        return "I am sorry, I ran into a technical problem. Please try again later."

# --- 5. Streamlit user interface ---
st.title("🏛️ City Hall Virtual Assistant")
st.caption(f"Using the model: {model}")

# Display the previous messages from the history
# We iterate over a copy to avoid issues if the list is modified during iteration
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- 6. Processing user input and generating answers ---
if prompt := st.chat_input("Ask your question here..."):
    # Add the user message to the internal history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Immediately display the user message in the interface
    with st.chat_message("user"):
        st.write(prompt)

    # Prepare the prompt with the recent history for the API
    prompt_messages_for_api = build_session_prompt(st.session_state.messages)

    # Display a loading indicator during generation
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Simple visual indicator

        # Generate the answer via the API
        response_content = generate_response(prompt_messages_for_api)

        # Display the full answer
        message_placeholder.write(response_content)

    # Add the assistant's answer to the internal history
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Optional: Add a button to clear the history
if st.button("Clear the conversation"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I am the City Hall virtual assistant. How can I help you today?"}]
    st.rerun() # Reload the page to show the initial state
