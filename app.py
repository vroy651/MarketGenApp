import os
import sys
import streamlit as st

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Import the Streamlit UI and authentication
from src.chatbot import main
from src.auth import init_auth, login, signup, is_authenticated, logout
from src.config import configure_streamlit_page

# Configure page first
configure_streamlit_page()

# Initialize authentication
init_auth()

# Run the application
if __name__ == "__main__":
    st.title("Pwani Content Generator")
    
    if not is_authenticated():
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            if login():
                st.rerun()
        
        with tab2:
            if signup():
                st.info("Please proceed to login.")
    else:
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()
        main()