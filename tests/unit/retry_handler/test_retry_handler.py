"""
Unit tests for DELM retry handler.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import traceback

from delm.utils.retry_handler import RetryHandler


class TestRetryHandler:
    """Test the RetryHandler class."""
    
    def test_initialization_defaults(self):
        """Test RetryHandler initialization with default values."""
        handler = RetryHandler()
        
        assert handler.max_retries == 3
        assert handler.base_delay == 1.0
    
    def test_initialization_custom(self):
        """Test RetryHandler initialization with custom values."""
        handler = RetryHandler(max_retries=5, base_delay=2.0)
        
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0
    
    def test_execute_with_retry_success_first_try(self):
        """Test successful execution on first try."""
        handler = RetryHandler(max_retries=3, base_delay=1.0)
    
        mock_func = Mock(return_value="success")
        mock_func.__name__ = "test_function"
    
        result = handler.execute_with_retry(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
    
    def test_execute_with_retry_success_after_retries(self):
        """Test successful execution after some retries."""
        handler = RetryHandler(max_retries=3, base_delay=0.1)  # Short delay for testing
    
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error1"), ValueError("error2"), "success"]
    
        result = handler.execute_with_retry(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_execute_with_retry_all_attempts_fail(self):
        """Test that the last exception is raised when all attempts fail."""
        handler = RetryHandler(max_retries=2, base_delay=0.1)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error1"), ValueError("error2"), RuntimeError("final error")]
        
        with pytest.raises(RuntimeError, match="final error"):
            handler.execute_with_retry(mock_func)
        
        assert mock_func.call_count == 3  # max_retries + 1
    
    def test_execute_with_retry_different_exceptions(self):
        """Test handling of different exception types."""
        handler = RetryHandler(max_retries=2, base_delay=0.1)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("value error"), TypeError("type error"), "success"]
        
        result = handler.execute_with_retry(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_execute_with_retry_exponential_backoff(self):
        """Test that delays increase exponentially."""
        handler = RetryHandler(max_retries=3, base_delay=1.0)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error1"), ValueError("error2"), ValueError("error3"), "success"]
        
        with patch('time.sleep') as mock_sleep:
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            assert mock_func.call_count == 4
            
            # Check that sleep was called with exponential delays
            expected_delays = [1.0, 2.0, 4.0]  # base_delay * 2^attempt
            assert mock_sleep.call_count == 3
            for i, delay in enumerate(expected_delays):
                assert mock_sleep.call_args_list[i][0][0] == delay
    
    def test_execute_with_retry_no_retries(self):
        """Test behavior when max_retries is 0."""
        handler = RetryHandler(max_retries=0, base_delay=1.0)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = ValueError("error")
        
        with pytest.raises(ValueError, match="error"):
            handler.execute_with_retry(mock_func)
        
        assert mock_func.call_count == 1  # Only one attempt
    
    def test_execute_with_retry_zero_base_delay(self):
        """Test behavior with zero base delay."""
        handler = RetryHandler(max_retries=2, base_delay=0.0)
    
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error1"), ValueError("error2"), "success"]
        
        with patch('time.sleep') as mock_sleep:
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            assert mock_func.call_count == 3
            
            # All delays should be 0
            expected_delays = [0.0, 0.0]
            assert mock_sleep.call_count == 2
            for i, delay in enumerate(expected_delays):
                assert mock_sleep.call_args_list[i][0][0] == delay
    
    def test_execute_with_retry_function_with_complex_args(self):
        """Test execution with complex function arguments."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        def complex_function(a, b, c=None, d=None, *args, **kwargs):
            return f"{a}_{b}_{c}_{d}_{args}_{kwargs}"
        
        mock_func = Mock(side_effect=[ValueError("error"), "success"])
        mock_func.__name__ = "test_function"
        
        result = handler.execute_with_retry(
            mock_func, 
            "arg1", 
            "arg2", 
            c="kwarg1", 
            d="kwarg2",
            extra_arg="extra",
            another_kwarg="another"
        )
        
        assert result == "success"
        assert mock_func.call_count == 2
        
        # Verify arguments were passed correctly
        expected_call = mock_func.call_args
        assert expected_call[0] == ("arg1", "arg2")
        assert expected_call[1] == {
            "c": "kwarg1", 
            "d": "kwarg2",
            "extra_arg": "extra",
            "another_kwarg": "another"
        }
    
    def test_execute_with_retry_preserves_exception_info(self):
        """Test that exception information is preserved."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        original_exception = ValueError("original error")
        mock_func.side_effect = [original_exception, "success"]
        
        result = handler.execute_with_retry(mock_func)
        
        assert result == "success"
        # The original exception should still be accessible
        assert original_exception.args[0] == "original error"
    
    def test_execute_with_retry_logging_behavior(self):
        """Test that logging occurs during retries."""
        handler = RetryHandler(max_retries=2, base_delay=0.1)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error1"), ValueError("error2"), "success"]
        
        with patch('delm.utils.retry_handler.log') as mock_log:
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            
            # Check that warnings were logged for failures
            warning_calls = [call for call in mock_log.warning.call_args_list if "Exception on attempt" in str(call)]
            assert len(warning_calls) == 2
            
            # Check that info was logged for retries
            info_calls = [call for call in mock_log.info.call_args_list if "Retrying in" in str(call)]
            assert len(info_calls) == 2
    
    def test_execute_with_retry_debug_logging(self):
        """Test debug logging behavior."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
    
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error"), "success"]
        
        with patch('delm.utils.retry_handler.log') as mock_log, \
             patch('delm.utils.retry_handler.traceback') as mock_traceback:
            
            # Mock that debug is enabled
            mock_log.isEnabledFor.return_value = True
            
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            
            # Check that traceback was printed for debug
            mock_traceback.print_exc.assert_called_once()
    
    def test_execute_with_retry_no_debug_logging(self):
        """Test that traceback is not printed when debug is disabled."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error"), "success"]
        
        with patch('delm.utils.retry_handler.log') as mock_log, \
             patch('delm.utils.retry_handler.traceback') as mock_traceback:
            
            # Mock that debug is disabled
            mock_log.isEnabledFor.return_value = False
            
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            
            # Check that traceback was not printed
            mock_traceback.print_exc.assert_not_called()
    
    def test_execute_with_retry_timing_logging(self):
        """Test that execution timing is logged."""
        handler = RetryHandler(max_retries=0, base_delay=0.1)
        
        mock_func = Mock(return_value="success")
        mock_func.__name__ = "test_function"
        
        with patch('delm.utils.retry_handler.log') as mock_log, \
             patch('time.time') as mock_time:
            
            # Mock time to return increasing values
            mock_time.side_effect = [100.0, 100.5]  # 0.5 second execution
            
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            
            # Check that timing was logged
            debug_calls = [call for call in mock_log.debug.call_args_list if "completed in" in str(call)]
            assert len(debug_calls) == 1
    
    def test_execute_with_retry_function_name_logging(self):
        """Test that function name is logged."""
        handler = RetryHandler(max_retries=0, base_delay=0.1)
        
        def test_function():
            return "success"
        
        mock_func = Mock(side_effect=test_function)
        mock_func.__name__ = "test_function"
        
        with patch('delm.utils.retry_handler.log') as mock_log:
            result = handler.execute_with_retry(mock_func)
            
            assert result == "success"
            
            # Check that function name was logged
            debug_calls = [call for call in mock_log.debug.call_args_list if "test_function" in str(call)]
            assert len(debug_calls) >= 1
    
    def test_execute_with_retry_error_logging_on_final_failure(self):
        """Test that error is logged when all attempts fail."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        mock_func = Mock()
        mock_func.__name__ = "test_function"
        mock_func.side_effect = [ValueError("error1"), ValueError("error2")]
        
        with patch('delm.utils.retry_handler.log') as mock_log:
            with pytest.raises(ValueError, match="error2"):
                handler.execute_with_retry(mock_func)
            
            # Check that error was logged
            error_calls = [call for call in mock_log.error.call_args_list if "All" in str(call)]
            assert len(error_calls) == 1
    
    def test_execute_with_retry_with_lambda_function(self):
        """Test execution with lambda function."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        lambda_func = lambda x, y: x + y
        
        mock_func = Mock(side_effect=lambda_func)
        mock_func.__name__ = "<lambda>"
        
        result = handler.execute_with_retry(mock_func, 2, 3)
        
        assert result == 5
        mock_func.assert_called_with(2, 3)
    
    def test_execute_with_retry_with_class_method(self):
        """Test execution with class method."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        class TestClass:
            def test_method(self, value):
                return f"processed_{value}"
        
        test_instance = TestClass()
        mock_func = Mock(side_effect=test_instance.test_method)
        mock_func.__name__ = "test_method"
        
        result = handler.execute_with_retry(mock_func, "test_value")
        
        assert result == "processed_test_value"
        mock_func.assert_called_with("test_value")
    
    def test_execute_with_retry_with_async_function_simulation(self):
        """Test execution with function that could be async (but we're not testing async)."""
        handler = RetryHandler(max_retries=1, base_delay=0.1)
        
        async def async_like_function(data):
            return f"async_result_{data}"
        
        # Create a mock that simulates the async function
        mock_func = Mock()
        mock_func.__name__ = "async_like_function"
        mock_func.side_effect = [ValueError("async error"), "async_result_test"]
        
        result = handler.execute_with_retry(mock_func, "test")
        
        assert result == "async_result_test"
        mock_func.assert_called_with("test") 