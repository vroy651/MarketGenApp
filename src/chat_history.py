from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import streamlit as st
from src.logger import logger

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict] = None

class ChatHistory:
    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.context_updates: Dict = {}
        self._initialize_session_state()
    
    @staticmethod
    def _initialize_session_state():
        """Initialize Streamlit session state for chat history"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'context_updates' not in st.session_state:
            st.session_state.context_updates = {}
        if 'campaign_data' not in st.session_state:
            st.session_state.campaign_data = {}
        if 'current_product' not in st.session_state:
            st.session_state.current_product = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a new message to the chat history"""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        st.session_state.chat_messages.append(message.dict())
        logger.info(f"Added new {role} message to chat history")
    
    def update_context(self, key: str, value: any) -> None:
        """Update context with new information during the chat"""
        self.context_updates[key] = {
            'value': value,
            'timestamp': datetime.now()
        }
        st.session_state.context_updates[key] = self.context_updates[key]
        logger.info(f"Updated context with key: {key}")
    
    def set_campaign_data(self, campaign_data: Dict) -> None:
        """Set the campaign data in the session state"""
        st.session_state.campaign_data = campaign_data
        logger.info("Campaign data initialized in session state")
        # Add initial system message with campaign context
        if campaign_data:
            system_message = f"Campaign Context:\n\nFresh Fri Campaign: {campaign_data.get('fresh_fri_campaign_summary', '')}\n\nPwani Oil Campaign Content:\n{campaign_data.get('pwani_oil_campaign_content', {}).get('body', '')}"
            self.add_message('system', system_message)
    
    def switch_product(self, product_data: Dict) -> None:
        """Switch to a different product while maintaining chat history"""
        st.session_state.current_product = product_data
        self.update_context('current_product', product_data)
        # Add a system message to acknowledge the product switch
        system_message = f"Switching context to: {product_data.get('name', 'New Product')}"
        self.add_message('system', system_message)
        logger.info(f"Switched to product: {product_data}")
    
    def get_current_product(self) -> Optional[Dict]:
        """Get the currently selected product"""
        return st.session_state.get('current_product')
    
    def get_campaign_data(self) -> Optional[Dict]:
        """Retrieve the campaign data from session state"""
        return st.session_state.get('campaign_data')
    
    def get_context(self, key: str) -> Optional[any]:
        """Retrieve context value by key"""
        context = self.context_updates.get(key, {})
        return context.get('value') if context else None
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get the most recent messages from chat history"""
        if not self.messages:
            return []
        start_idx = max(0, len(self.messages) - limit)
        return self.messages[start_idx:]
    
    def get_full_context(self) -> Dict:
        """Get the complete chat context including messages and updates"""
        return {
            'messages': [msg.dict() for msg in self.messages],
            'context_updates': self.context_updates,
            'campaign_data': st.session_state.get('campaign_data', {}),
            'current_product': st.session_state.get('current_product')
        }
    
    def clear_history(self) -> None:
        """Clear the chat history and context updates"""
        # Clear internal message list
        self.messages.clear()
        
        # Clear context updates
        self.context_updates.clear()
        
        # Clear Streamlit session state
        st.session_state.chat_messages = []
        st.session_state.context_updates = {}
        st.session_state.campaign_data = {}
        st.session_state.current_product = None
        
        # Reset all session state variables
        if 'chat_messages' in st.session_state:
            st.session_state.chat_messages.clear()
        if 'context_updates' in st.session_state:
            st.session_state.context_updates.clear()
        if 'campaign_data' in st.session_state:
            st.session_state.campaign_data.clear()
        
        # Reset current product
        if 'current_product' in st.session_state:
            st.session_state.current_product = None
        
        logger.info("Chat history and all related data cleared successfully")
    
    def delete_message(self, index: int) -> None:
        """Delete a message from the chat history by index"""
        if 0 <= index < len(self.messages):
            self.messages.pop(index)
            st.session_state.chat_messages.pop(index)
            logger.info(f"Deleted message at index {index} from chat history")
        else:
            logger.warning(f"Invalid index {index} for message deletion")