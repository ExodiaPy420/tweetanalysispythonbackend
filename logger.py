"""
Logging configuration for the sentiment analysis API.
"""
import logging
import sys
import os
from datetime import datetime


def setup_logger(name: str = "sentiment_api") -> logging.Logger:
    """
    Configure application logger with console and file handlers.
    
    Args:
        name: Logger name (default: "sentiment_api")
        
    Returns:
        Configured logger instance
    """
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(
        f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
