import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mock dependencies that might be missing in the environment
sys.modules['dotenv'] = MagicMock()
sys.modules['keras_nlp'] = MagicMock()
# Also mock submodules if needed
sys.modules['keras_nlp.models'] = MagicMock()

# Add the parent directory to sys.path so we can import gemma_service
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now import the module to be tested
# Using importlib to ensure we get a fresh import if needed,
# though for a script execution it's usually fine.
from gemma_service import gemma_model

class TestGemmaModel(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test if necessary
        pass

    @patch('gemma_service.gemma_model.load_dotenv')
    @patch('gemma_service.gemma_model.os.getenv')
    def test_initialize_model_success(self, mock_getenv, mock_load_dotenv):
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
