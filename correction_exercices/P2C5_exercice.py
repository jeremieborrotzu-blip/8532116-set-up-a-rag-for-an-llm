# tests/test_database.py
import unittest
from unittest.mock import patch, MagicMock
import datetime


# Import the functions to test
from utils.database import log_interaction, log_feedback, Interaction # Make sure Interaction is importable


class TestDatabaseFunctions(unittest.TestCase):


    @patch('utils.database.SessionLocal') # Patch SessionLocal where it is defined/imported
    def test_log_interaction_success(self, MockSessionLocal):
        """Checks that log_interaction calls the right session methods."""
        # Configure the mock of the session and its methods
        mock_db_session = MagicMock()
        MockSessionLocal.return_value = mock_db_session


        # Simulate the created Interaction object and the return of db.refresh()
        mock_interaction_instance = MagicMock(spec=Interaction)
        mock_interaction_instance.id = 123 # Simulate a returned ID
        def refresh_side_effect(instance):
            # Simulate the assignment of the ID by the DB during the refresh
            instance.id = 123
        mock_db_session.refresh.side_effect = refresh_side_effect


        # Sample data
        test_query = "What time is it?"
        test_contexts = ["Context 1", "Context 2"]
        test_response = "It's time to code!"


        # Call the function
        returned_id = log_interaction(test_query, test_contexts, test_response)


        # Assertions
        # 1. Was SessionLocal() called to get a session?
        MockSessionLocal.assert_called_once()


        # 2. Was db.add called with an Interaction object?
        self.assertEqual(mock_db_session.add.call_count, 1)
        added_object = mock_db_session.add.call_args[0][0] # Get the object passed to add
        self.assertIsInstance(added_object, Interaction)
        self.assertEqual(added_object.user_query, test_query)
        self.assertEqual(added_object.llm_response, test_response)
        self.assertEqual(added_object.contexts, test_contexts)


        # 3. Was db.commit called?
        mock_db_session.commit.assert_called_once()


        # 4. Was db.refresh called (to get the ID)?
        mock_db_session.refresh.assert_called_once_with(added_object)


        # 5. Does the function return the simulated ID?
        self.assertEqual(returned_id, 123)


        # 6. Was db.close called?
        mock_db_session.close.assert_called_once()


    @patch('utils.database.SessionLocal')
    def test_log_feedback_updates_score(self, MockSessionLocal):
      """Checks that log_feedback updates the interaction score."""
      # Configure the mock of the session
      mock_db_session = MagicMock()
      MockSessionLocal.return_value = mock_db_session


      # Simulate the Interaction object found by the query
      mock_interaction_found = MagicMock(spec=Interaction)
      mock_interaction_found.id = 456
      mock_interaction_found.feedback_score = None # Initial score


      # Configure query().filter().first() to return our mock
      mock_db_session.query.return_value.filter.return_value.first.return_value = mock_interaction_found


      # Sample data
      test_interaction_id = 456
      test_score = 1 # Positive feedback


      # Call the function
      log_feedback(test_interaction_id, test_score)


      # Assertions
      # 1. Was query(Interaction).filter(Interaction.id == ...).first() called?
      mock_db_session.query.assert_called_once_with(Interaction)
      mock_db_session.query.return_value.filter.assert_called_once() # Check the call to filter
      mock_db_session.query.return_value.filter.return_value.first.assert_called_once()


      # 2. Was the score of the found Interaction object updated?
      self.assertEqual(mock_interaction_found.feedback_score, test_score)


      # 3. Was db.commit called?
      mock_db_session.commit.assert_called_once()


      # 4. Was db.close called?
      mock_db_session.close.assert_called_once()




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
