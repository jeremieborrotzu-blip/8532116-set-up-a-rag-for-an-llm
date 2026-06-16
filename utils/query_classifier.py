"""
Query classification module to determine whether a question requires RAG
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.config import MISTRAL_API_KEY, CHAT_MODEL, COMMUNE_NAME

class QueryClassifier:
    """
    Class to classify queries and determine whether they require RAG
    """

    def __init__(self):
        """
        Initializes the query classifier
        """
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

        # Town-related keywords that suggest a need for RAG
        self.commune_keywords = [
            COMMUNE_NAME.lower(),
            "city hall", "town", "city", "municipal", "municipality",
            "council", "mayor", "deputy", "official", "service",
            "hours", "opening", "closing", "address", "contact",
            "document", "form", "procedure", "administrative",
            "zoning", "permit", "construction", "works",
            "school", "daycare", "nursery", "cafeteria", "education",
            "association", "sport", "culture", "leisure", "library",
            "event", "gathering", "festival", "market",
            "transport", "bus", "traffic", "parking", "parking lot",
            "waste", "trash", "recycling", "environment",
            "tax", "levy", "budget", "finance"
        ]

        # General questions that do not require RAG
        self.general_patterns = [
            r"^(hi|hello|hey|good morning|good evening)[\s\.,!]*$",
            r"^(thanks|thank you|thank you very much)[\s\.,!]*$",
            r"^(how are you|how's it going|how do you do)[\s\.,!?]*$",
            r"^(goodbye|bye|see you|see you later|see you soon)[\s\.,!]*$",
            r"^(who are you|what are you|what do you do|how do you work)[\s\?]*$",
            r"^(help|sos|i need help)[\s\.,!?]*$"
        ]

    def needs_rag(self, query: str) -> Tuple[bool, float, str]:
        """
        Determines whether a query requires RAG

        Args:
            query: User query

        Returns:
            Tuple (needs_rag, confidence, reason)
        """
        # Convert the query to lowercase for comparison
        query_lower = query.lower()

        # 1. Check the general question patterns (greetings, thanks, etc.)
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return False, 0.95, "General question or greeting"

        # 2. Check for the presence of town-related keywords
        commune_keywords_found = [kw for kw in self.commune_keywords if kw in query_lower]
        if commune_keywords_found:
            keywords_str = ", ".join(commune_keywords_found)
            return True, 0.9, f"Contains town-related keywords: {keywords_str}"

        # 3. Use the LLM for ambiguous cases
        if self.mistral_client:
            return self._classify_with_llm(query)

        # By default, use RAG for long questions (more than 5 words)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Complex question (more than 5 words)"

        # By default, do not use RAG
        return False, 0.5, "No specific criterion detected"

    def _classify_with_llm(self, query: str) -> Tuple[bool, float, str]:
        """
        Uses the LLM to classify the query

        Args:
            query: User query

        Returns:
            Tuple (needs_rag, confidence, reason)
        """
        try:
            system_prompt = f"""You are a query classifier for a virtual assistant for the town of {COMMUNE_NAME}.
Your task is to determine whether a question requires a search in a knowledge base specific to the town.

Answer ONLY with "RAG" or "DIRECT" followed by a brief explanation:
- "RAG" if the question is about information specific to {COMMUNE_NAME} (municipal services, events, addresses, hours, etc.)
- "DIRECT" if it is a general question, a greeting, or a question that does not require information specific to the town.

Examples:
Question: "Hello, how are you?"
Answer: DIRECT - Simple greeting

Question: "What are the City Hall opening hours?"
Answer: RAG - Request for information specific to the town

Question: "Who is the current mayor?"
Answer: RAG - Request for information specific to the town

Question: "What is artificial intelligence?"
Answer: DIRECT - General knowledge question
"""

            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query)
            ]

            response = self.mistral_client.chat(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent answers
                max_tokens=50  # A short answer is enough
            )

            result = response.choices[0].message.content.strip()
            logging.info(f"LLM classification for '{query}': {result}")

            # Parse the answer
            if result.startswith("RAG"):
                confidence = 0.85  # High confidence in the LLM decision
                reason = result.replace("RAG - ", "").replace("RAG-", "").replace("RAG:", "").strip()
                return True, confidence, reason
            elif result.startswith("DIRECT"):
                confidence = 0.85
                reason = result.replace("DIRECT - ", "").replace("DIRECT-", "").replace("DIRECT:", "").strip()
                return False, confidence, reason
            else:
                # Ambiguous answer, use RAG by default
                return True, 0.6, "Ambiguous classification, using RAG as a precaution"

        except Exception as e:
            logging.error(f"Error during the LLM classification: {e}")
            # On error, use RAG by default
            return True, 0.5, f"Classification error: {str(e)}"
