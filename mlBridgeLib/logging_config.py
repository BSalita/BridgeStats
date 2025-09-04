"""
Centralized logging configuration for ML Bridge project.
Provides consistent logging with timestamp, function name, and log level.
"""

import logging
import inspect
from datetime import datetime
from functools import wraps
import sys
import os

# Python's logging system automatically captures function names when called properly

def setup_logger(name=None, level=logging.INFO, log_file=None):
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (defaults to calling module name)
        level: Logging level (default: INFO)
        log_file: Optional file to log to (in addition to console)
    
    Returns:
        logger: Configured logger instance
    """
    if name is None:
        # Get the name of the calling module
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'ml_bridge')
    
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter with timestamp, function name, level, and message
    # Python's logging automatically detects function names when called properly
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent messages from being passed to ancestor loggers (avoids duplicates)
    logger.propagate = False
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name=None):
    """
    Get or create a logger for the calling module.
    
    Args:
        name: Logger name (defaults to calling module name)
    
    Returns:
        logger: Logger instance
    """
    if name is None:
        # Get the name of the calling module
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'ml_bridge')
    
    return logging.getLogger(name)

# Convenience function to replace print statements
def log_print(*args, level=logging.INFO, sep=' ', end=''):
    """
    Drop-in replacement for print() that uses logging.
    
    Args:
        *args: Arguments to print/log
        level: Logging level (default: INFO)
        sep: Separator between arguments (default: space)
        end: End character (ignored for logging)
    """
    # Get logger for the calling module
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'ml_bridge')
    logger = get_logger(module_name)
    
    # Convert args to string like print() would
    message = sep.join(str(arg) for arg in args)
    
    # Get the calling function name
    caller_name = frame.f_code.co_name
    
    # Use the standard logging methods which automatically capture function names
    if level == logging.DEBUG:
        logger.debug(message)
    elif level == logging.INFO:
        logger.info(message)
    elif level == logging.WARNING:
        logger.warning(message)
    elif level == logging.ERROR:
        logger.error(message)
    else:
        logger.log(level, message)

# Initialize default logger for the project
def init_project_logging(log_file=None, level=logging.INFO):
    """
    Initialize logging for the entire ML Bridge project.
    
    Args:
        log_file: Optional log file path
        level: Default logging level
    """
    return setup_logger('ml_bridge', level=level, log_file=log_file)
