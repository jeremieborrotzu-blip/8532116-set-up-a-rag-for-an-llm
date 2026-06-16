# utils/query_classifier.py
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# It can be useful to define the possible intents as constants
INTENT_RAG = "RAG"
INTENT_CHAT = "CHAT"
DEFAULT_INTENT = INTENT_RAG # Choose RAG by default to favor the search


def classify_query_intent(query: str, client: MistralClient, model: str = "mistral-large-latest") -> str:
    """
    Classifies the intent of the user query using the Mistral API.


    Args:
        query: The question asked by the user.
        client: The initialized Mistral client.
        model: The Mistral model to use for the classification.


    Returns:
        The detected intent ("RAG" or "CHAT").
    """
    classification_system_prompt = f"""
    Your role is to classify the intent of the user's question for a City Hall chatbot.
    Answer only with "RAG" or "CHAT". Do not provide any other explanation.


    - Answer "RAG" if the question is looking for specific information that could be found in the City Hall documents (administrative procedures, hours, required documents, regulations, municipal services, specific local information).
    - Answer "CHAT" if the question is a greeting, a polite formula, general conversation, a question off-topic for City Hall, or a simple social interaction.


    Examples:
    - "What documents do I need for a passport?" -> RAG
    - "Hello, how are you?" -> CHAT
    - "What are the hours of the municipal swimming pool?" -> RAG
    - "Thanks!" -> CHAT
    - "Tell me about tomorrow's weather" -> CHAT
    - "How do I enroll my child in school?" -> RAG


    Question to classify:
    """


    messages = [
        ChatMessage(role="system", content=classification_system_prompt),
        ChatMessage(role="user", content=query)
    ]


    try:
        logging.info(f"Classifying the query: '{query[:50]}...'")
        response = client.chat(
            model=model,
            messages=messages,
            temperature=0.1, # Low temperature for a more deterministic answer
            max_tokens=5     # Very short, we only expect RAG or CHAT
        )
        intent = response.choices[0].message.content.strip().upper()


        if intent == INTENT_RAG:
            logging.info(f"Detected intent: {INTENT_RAG}")
            return INTENT_RAG
        elif intent == INTENT_CHAT:
            logging.info(f"Detected intent: {INTENT_CHAT}")
            return INTENT_CHAT
        else:
            logging.warning(f"Unclear classification received: '{intent}'. Using the default intent: {DEFAULT_INTENT}")
            return DEFAULT_INTENT # Return the default intent if the answer is not clear


    except Exception as e:
        logging.error(f"Error while classifying the query: {e}")
        return DEFAULT_INTENT # Return the default intent on error
