# utils/database.py
import os
import datetime
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from .config import DATABASE_URL, DATABASE_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the database folder if it does not exist
os.makedirs(DATABASE_DIR, exist_ok=True)

# Create the SQLAlchemy engine for the SQLite database
# `check_same_thread=False` is required for SQLite with Streamlit/multithreading
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False) # echo=True to see the SQL queries

# Create a declarative base for the ORM models
Base = declarative_base()

# Define the ORM model for the interactions table
class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    query = Column(Text, nullable=False)
    response = Column(Text)
    sources = Column(JSON) # Stores the list of source dictionaries as JSON
    query_metadata = Column(JSON) # Stores the metadata (mode, confidence, etc.)
    feedback = Column(String(20)) # e.g. "👍", "👎"
    feedback_value = Column(Integer) # 1 for positive, 0 for negative, NULL for none
    feedback_comment = Column(Text) # Optional: feedback comment

# Create the table in the database if it does not already exist
try:
    Base.metadata.create_all(engine)
    logging.info("Table 'interactions' checked/created in the database.")
except SQLAlchemyError as e:
    logging.error(f"Error while creating/checking the table: {e}")

# Create a session factory to interact with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Utility function to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def log_interaction(query: str, response: str, sources: list, metadata: dict = None, feedback: str = None, feedback_comment: str = None):
    """Logs an interaction in the database.

    Args:
        query: User question
        response: Generated response
        sources: List of sources used
        metadata: Metadata (mode, confidence, etc.)
        feedback: User feedback
        feedback_comment: Feedback comment

    Returns:
        ID of the logged interaction
    """
    db_session = SessionLocal()
    try:
        interaction = Interaction(
            query=query,
            response=response,
            sources=sources, # SQLAlchemy handles the JSON serialization
            query_metadata=metadata, # Metadata (mode, confidence, etc.)
            feedback=feedback,
            feedback_comment=feedback_comment
        )
        db_session.add(interaction)
        db_session.commit()

        # Log with information about the mode used
        mode_info = ""
        if metadata and "mode" in metadata:
            mode_info = f", Mode: {metadata['mode']}"

        logging.info(f"Interaction logged (Query: '{query[:50]}...'{mode_info}, Feedback: {feedback})")
        return interaction.id # Return the ID of the logged interaction
    except SQLAlchemyError as e:
        logging.error(f"Error while logging the interaction: {e}")
        db_session.rollback() # Roll back the changes on error
        return None
    finally:
        db_session.close() # Always close the session

def get_all_interactions(limit: int = 100):
    """Retrieves the latest interactions from the database."""
    db_session = SessionLocal()
    try:
        interactions = db_session.query(Interaction).order_by(Interaction.timestamp.desc()).limit(limit).all()
        logging.info(f"{len(interactions)} interactions retrieved.")
        # Converts the Interaction objects to dictionaries for easier manipulation (e.g. Pandas)
        return [
            {
                "id": inter.id,
                "timestamp": inter.timestamp,
                "query": inter.query,
                "response": inter.response,
                "sources": inter.sources, # Already a list of dicts (or None)
                "metadata": inter.query_metadata, # Metadata (mode, confidence, etc.)
                "feedback": inter.feedback,
                "feedback_comment": inter.feedback_comment,
            }
            for inter in interactions
        ]
    except SQLAlchemyError as e:
        logging.error(f"Error while retrieving the interactions: {e}")
        return []
    finally:
        db_session.close()

def update_feedback(interaction_id: int, feedback: str, feedback_comment: str = None, feedback_value: int = None):
    """Updates the feedback for a specific interaction.

    Args:
        interaction_id: ID of the interaction to update
        feedback: Feedback text (emoji)
        feedback_comment: Optional comment
        feedback_value: Numeric value (1 for positive, 0 for negative)

    Returns:
        True if the update succeeded, False otherwise
    """
    db_session = SessionLocal()
    try:
        interaction = db_session.query(Interaction).filter(Interaction.id == interaction_id).first()
        if interaction:
            # Update the values
            interaction.feedback = feedback
            interaction.feedback_value = feedback_value
            interaction.feedback_comment = feedback_comment

            # Save the changes
            db_session.commit()
            logging.info(f"Feedback updated for interaction ID {interaction_id}")
            return True
        else:
            logging.warning(f"Interaction ID {interaction_id} not found for the feedback update.")
            return False
    except SQLAlchemyError as e:
        logging.error(f"Error while updating the feedback for interaction {interaction_id}: {e}")
        db_session.rollback()
        return False
    finally:
        db_session.close()
