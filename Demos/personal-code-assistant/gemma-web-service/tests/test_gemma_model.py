import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Pylint directive to ignore import errors for mocked modules
# pylint: disable=import-error
# Pylint directive to allow imports not at top of file
# pylint: disable=wrong-import-position

# Mock dependencies that might be missing in the environment or need isolation
# We must mock these before importing gemma_model because it imports them at top-level.
sys.modules['dotenv'] = MagicMock()
sys.modules['keras_nlp'] = MagicMock()
# Also mock submodules if needed for deep imports
sys.modules['keras_nlp.models'] = MagicMock()

# Add the parent directory to sys.path so we can import gemma_service
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the module to be tested
# Using strict import to ensure we get the module from our path
try:
    from gemma_service import gemma_model
except ImportError:
    # Fallback if package structure is not standard
    import gemma_model

class TestGemmaModel(unittest.TestCase):
    """Test suite for gemma_model.py initialization logic."""

    def setUp(self):
        """Setup for each test."""
        # Reset mocks before each test if necessary
        pass

    @patch('gemma_service.gemma_model.load_dotenv')
    @patch('gemma_service.gemma_model.os.getenv')
    def test_initialize_model_success(self, mock_getenv, mock_load_dotenv):
        """Test successful initialization with valid environment variables."""
        # Setup mocks
        mock_getenv.side_effect = lambda key: {
            'KAGGLE_USERNAME': 'test_user',
            'KAGGLE_KEY': 'test_key'
        }.get(key, None)

        # Mock the model class inside gemma_model module
        # Since we mocked keras_nlp in sys.modules, gemma_model.keras_nlp is that mock.
        # We need to configure it to return our expected model instance.
        mock_keras_nlp = gemma_model.keras_nlp
        mock_model_class = mock_keras_nlp.models.GemmaCausalLM
        mock_model_instance = MagicMock()
        mock_model_class.from_preset.return_value = mock_model_instance

        # Call the function
        result = gemma_model.initialize_model()

        # Assertions
        mock_load_dotenv.assert_called_once()
        mock_model_class.from_preset.assert_called_once_with(gemma_model.gemma_model_id)
        self.assertEqual(result, mock_model_instance)

    @patch('gemma_service.gemma_model.load_dotenv')
    @patch('gemma_service.gemma_model.os.getenv')
    def test_initialize_model_missing_username(self, mock_getenv, mock_load_dotenv):
        """Test initialization failure when KAGGLE_USERNAME is missing."""
        # Setup mocks
        mock_getenv.side_effect = lambda key: {
            'KAGGLE_USERNAME': None,
            'KAGGLE_KEY': 'test_key'
        }.get(key, None)

        # Call the function and assert exception
        with self.assertRaisesRegex(ValueError, "KAGGLE_USERNAME environment variable not found"):
            gemma_model.initialize_model()

    @patch('gemma_service.gemma_model.load_dotenv')
    @patch('gemma_service.gemma_model.os.getenv')
    def test_initialize_model_missing_key(self, mock_getenv, mock_load_dotenv):
        """Test initialization failure when KAGGLE_KEY is missing."""
        # Setup mocks
        mock_getenv.side_effect = lambda key: {
            'KAGGLE_USERNAME': 'test_user',
            'KAGGLE_KEY': None
        }.get(key, None)

        # Call the function and assert exception
        with self.assertRaisesRegex(ValueError, "KAGGLE_KEY environment variable not found"):
            gemma_model.initialize_model()

if __name__ == '__main__':
    unittest.main()
