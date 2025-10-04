"""
Unit tests for logger configuration.

Tests the custom logger setup including basic functionality and proper 
integration with the existing codebase. These tests work with pytest's
logging capture system.
"""

import logging

from iceaggr.utils import get_logger


class TestCustomLogger:
    """Test suite for custom logger configuration."""

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a proper Logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_default_level_is_info(self):
        """Test that default logging level is INFO."""
        logger = get_logger("test_default_level")
        assert logger.level == logging.INFO

    def test_logger_custom_level(self):
        """Test that custom logging level can be set."""
        logger = get_logger("test_custom_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_level_can_be_updated(self):
        """Test that logging level can be updated on existing logger."""
        logger = get_logger("test_update_level", level=logging.INFO)
        assert logger.level == logging.INFO
        
        logger = get_logger("test_update_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_does_not_propagate(self):
        """Test that logger does not propagate to root logger."""
        logger = get_logger("test_propagate")
        assert logger.propagate is False

    def test_logger_same_name_returns_same_instance(self):
        """Test that requesting the same logger name returns the same instance."""
        logger1 = get_logger("test_same_instance")
        logger2 = get_logger("test_same_instance")
        assert logger1 is logger2

    def test_logger_can_log_at_different_levels(self):
        """Test that logger can be configured with different log levels."""
        # Test that logger works at INFO level (default)
        logger_info = get_logger("test_logging_info", level=logging.INFO)
        assert logger_info.level == logging.INFO
        assert logger_info.isEnabledFor(logging.INFO)
        assert logger_info.isEnabledFor(logging.WARNING)
        assert logger_info.isEnabledFor(logging.ERROR)
        assert not logger_info.isEnabledFor(logging.DEBUG)
        
        # Test that logger works at DEBUG level
        logger_debug = get_logger("test_logging_debug", level=logging.DEBUG)
        assert logger_debug.level == logging.DEBUG
        assert logger_debug.isEnabledFor(logging.DEBUG)
        assert logger_debug.isEnabledFor(logging.INFO)
        
        # Test that logger works at WARNING level
        logger_warn = get_logger("test_logging_warn", level=logging.WARNING)
        assert logger_warn.level == logging.WARNING
        assert logger_warn.isEnabledFor(logging.WARNING)
        assert not logger_warn.isEnabledFor(logging.INFO)
        assert not logger_warn.isEnabledFor(logging.DEBUG)

    def test_different_loggers_are_independent(self):
        """Test that different logger names create independent loggers."""
        logger1 = get_logger("test_independent_1", level=logging.DEBUG)
        logger2 = get_logger("test_independent_2", level=logging.WARNING)
        
        assert logger1 is not logger2
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.WARNING

    def test_logger_with_module_name(self):
        """Test that logger works with __name__ pattern."""
        logger = get_logger(__name__)
        assert logger.name == __name__
        assert isinstance(logger, logging.Logger)


class TestLoggerIntegration:
    """Test integration of logger with existing codebase."""

    def test_logger_can_be_imported_from_main_package(self):
        """Test that get_logger can be imported from main iceaggr package."""
        from iceaggr import get_logger as imported_logger
        
        logger = imported_logger("test_import")
        assert isinstance(logger, logging.Logger)

    def test_logger_can_be_imported_from_utils(self):
        """Test that get_logger can be imported from utils module."""
        from iceaggr.utils import get_logger as utils_logger
        
        logger = utils_logger("test_utils_import")
        assert isinstance(logger, logging.Logger)

    def test_dataset_module_uses_logger(self):
        """Test that dataset module has logger configured."""
        from iceaggr.data import dataset
        
        # Check that the module has a logger attribute
        assert hasattr(dataset, 'logger')
        assert isinstance(dataset.logger, logging.Logger)

    def test_logger_does_not_interfere_with_pytest(self):
        """Test that our logger doesn't interfere with pytest's logging."""
        logger = get_logger("test_pytest_compat")
        
        # This should not raise any errors or warnings
        logger.info("Test message during pytest")
        
        # Pytest's caplog should still work
        assert True  # If we get here, no interference occurred
