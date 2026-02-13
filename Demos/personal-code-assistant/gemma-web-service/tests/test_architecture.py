
import unittest
import sys
import inspect
import os
from unittest.mock import MagicMock, patch

class TestArchitecture(unittest.TestCase):
    def test_process_text_is_sync(self):
        """
        Verify that process_text is a synchronous function.
        Blocking model inference in an async function blocks the event loop.
        By making it synchronous, FastAPI runs it in a threadpool, unblocking the loop.
        """
        # Create mocks
        mock_fastapi = MagicMock()
        mock_app_instance = MagicMock()

        # Decorators that return the function unchanged
        def route_decorator(path):
            def decorator(func):
                return func
            return decorator

        mock_app_instance.post.side_effect = route_decorator
        mock_app_instance.get.side_effect = route_decorator
        mock_fastapi.FastAPI.return_value = mock_app_instance

        # Dictionary of modules to mock
        modules_to_patch = {
            'fastapi': mock_fastapi,
            'uvicorn': MagicMock(),
            'pydantic': MagicMock(),
            'gemma_model': MagicMock(),
        }

        # Apply patches to sys.modules
        with patch.dict(sys.modules, modules_to_patch):
            # Add source directory to path if not already there
            current_dir = os.path.dirname(os.path.abspath(__file__))
            service_dir = os.path.join(current_dir, '../gemma_service')
            if service_dir not in sys.path:
                sys.path.append(service_dir)

            # Import module inside the patched context
            # ensure we get a fresh import
            if 'gemma_service_main' in sys.modules:
                del sys.modules['gemma_service_main']

            try:
                import gemma_service_main
            except ImportError as e:
                self.fail(f"Failed to import gemma_service_main: {e}")

            # Check if process_text is defined
            self.assertTrue(hasattr(gemma_service_main, 'process_text'), "process_text not found in module")

            # Check if it is a coroutine function (async def)
            # We expect it to NOT be a coroutine function for better performance
            is_coroutine = inspect.iscoroutinefunction(gemma_service_main.process_text)

            if is_coroutine:
                self.fail("process_text is defined as 'async def'. Change to 'def' to unblock the event loop during model inference.")
            else:
                self.assertFalse(is_coroutine, "process_text should be synchronous")

if __name__ == '__main__':
    unittest.main()
