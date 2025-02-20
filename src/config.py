# config.py
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from typing import Tuple, Optional

# Load environment variables
load_dotenv()

# Get logger
logger = logging.getLogger(__name__)

# --- Constants for Branding ---
PRIMARY_COLOR = "#FF4B4B"  # Streamlit default red
SECONDARY_COLOR = "#F0F2F6"  # Streamlit default background
BACKGROUND_COLOR = "#FFFFFF"  # White background
TEXT_COLOR = "#31333F"  # Streamlit default text color
INPUT_BG_COLOR = "#FFFFFF"  # Clean white for input fields
INPUT_BORDER_COLOR = "#CCCCCC"  # Streamlit default border color

# Environment and configuration validation
REQUIRED_ENV_VARS = [
    'APP_ENV',
    'LOG_LEVEL',
    'ENABLE_RATE_LIMITING',
    'MAX_REQUESTS_PER_MINUTE',
    'DEFAULT_MODEL',
    'MODEL_TEMPERATURE',
    'TAVILY_API_KEY'  # Added Tavily API key requirement
]

def validate_environment() -> None:
    """Validate all required environment variables are set."""
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Validate specific values
    try:
        if not 0 <= float(os.getenv('MODEL_TEMPERATURE', '0')) <= 1:
            raise ValueError("MODEL_TEMPERATURE must be between 0 and 1")

        if not 0 < int(os.getenv('MAX_REQUESTS_PER_MINUTE', '0')):
            raise ValueError("MAX_REQUESTS_PER_MINUTE must be greater than 0")

        # Validate Tavily API key format
        tavily_key = os.getenv('TAVILY_API_KEY')
        if not tavily_key or not tavily_key.startswith('tvly-'):
            raise ValueError("TAVILY_API_KEY must be a valid Tavily API key starting with 'tvly-'")
    except ValueError as e:
        raise EnvironmentError(f"Invalid environment variable value: {str(e)}")

# Configure Streamlit page
def configure_streamlit_page():
    st.set_page_config(
        page_title="Pwani Oil Marketing Generator",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# Load API Keys
def load_api_keys():
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not google_api_key and not openai_api_key:
        st.error("🔑 No API Keys found.  Please set either GOOGLE_API_KEY or OPENAI_API_KEY in your .env file")
        st.stop()

    return google_api_key, openai_api_key  # Modified to return only two values

# Custom CSS for better styling and branding
def load_css():
    # Function kept for compatibility, but CSS removed to use Streamlit's default styling
    pass