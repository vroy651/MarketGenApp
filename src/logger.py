import logging
import logging.handlers
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get log level from environment variable
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, LOG_LEVEL, logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(log_level)

# Console handler with configurable level
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# File handler with detailed formatting for debugging
file_handler = logging.handlers.RotatingFileHandler(
    log_dir / 'app.log',
    maxBytes=50*1024*1024,  # 50MB
    backupCount=10,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(message)s'
)
file_handler.setFormatter(file_formatter)

# Add handlers to root logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Create specific loggers for different components with error tracking
component_loggers = {}
for component in ['rag', 'workflow', 'ui', 'chatbot', 'llm']:
    component_logger = logging.getLogger(component)
    component_logger.setLevel(log_level)
    component_loggers[component] = component_logger

# Export the loggers
__all__ = ['logger'] + list(component_loggers.keys())

# Add exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the exception handler
import sys
sys.excepthook = handle_exception