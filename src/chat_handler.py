import streamlit as st
import logging
import random
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from src.logger import logger
import re
class SocialMediaContent(BaseModel):
    platform: str
    content: str
    hashtags: Optional[list[str]] = None
    media_type: Optional[str] = None

class EmailContent(BaseModel):
    subject: str
    body: str
    call_to_action: Optional[str] = None

class MarketingContent(BaseModel):
    campaign_name: str
    target_audience: str
    key_messages: list[str]
    channels: list[str]
    content: str

class ChatbotOutput(BaseModel):
    """Pydantic model to structure the chatbot's output in a more conversational way."""
    role: str = Field("assistant", description="The role of the message sender (user or assistant)")
    content_type: str = "text"  # Type of content (text, social_media, email, marketing, summary)
    text_content: Optional[str] = Field(None, description="General text content, for simple responses. give only text for the product not itroduction or any extra information")
    suggested_questions: Optional[List[str]] = Field(None, description="Follow-up questions to maintain conversation")
    campaign_context: Optional[Dict] = Field(None, description="Current campaign context")
    social_media_content: Optional[SocialMediaContent] = None
    email_content: Optional[EmailContent] = None
    marketing_content: Optional[MarketingContent] = None

    @classmethod
    def from_text(cls, text: str) -> 'ChatbotOutput':
        return cls(role="assistant", content_type="text", text_content=text)

    def dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content_type": self.content_type,
            "text_content": self.text_content,
            "social_media_content": self.social_media_content.dict() if self.social_media_content else None,
            "email_content": self.email_content.dict() if self.email_content else None,
            "marketing_content": self.marketing_content.dict() if self.marketing_content else None
        }

    # def render_text(self) -> str:
    #     """Renders the output in a conversational format."""
    #     output_parts = []
        
    #     # Add main content
    #     if self.text_content:
    #         output_parts.append(self.convert_json_to_natural_language(self.text_content))
        
    #     # Add a natural follow-up if available
    #     if self.suggested_questions:
    #         # Only add one follow-up question randomly
    #         follow_up = random.choice(self.suggested_questions)
    #         output_parts.append(f"\n\n{follow_up}")
        
    #     return "\n".join(output_parts)

    def render_text(self) -> str:
        """Renders only the requested content without any additional information."""
        if self.text_content:
            # Return only the content, without any follow-up questions
            return self.text_content.strip()
        return ""

    def render(self) -> str:
        return self.render_text()

    def clean_response_format(response: str) -> str:
        """Cleans up the response to ensure only the marketing content remains."""
        if not isinstance(response, str):
            return str(response)

        # Remove any JSON-like formatting
        response = response.strip()
        if response.startswith('{') or response.startswith('['):
            try:
                import json
                data = json.loads(response)
                if isinstance(data, dict):
                    response = data.get('response', '') or data.get('text', '') or data.get('content', '') or str(data)
                elif isinstance(data, list):
                    response = '\n'.join(str(item) for item in data if item)
            except:
                # If JSON parsing fails, remove basic JSON markers
                response = re.sub(r'^[{\[]|[}\]]$', '', response)
                response = response.replace('"response":', '').replace('"text":', '').replace('"content":', '')

        # Remove any remaining quotes and clean up
        response = response.replace('"', '').strip()

        # Strip out common introductory phrases
        intro_patterns = [
            r"^(Here'?s?( a)?|I'?ve created|Based on|According to|Here is|This is|Try this|How about).*?campaign.*?[:,]",
            r"^(Here'?s?( a)?|I'?ve created|Based on|According to|Here is|This is|Try this|How about).*?content.*?[:,]",
            r"^(Here'?s?( a)?|I'?ve created|Based on|According to|Here is|This is|Try this|How about).*?message.*?[:,]",
            r"^Okay,? so you want.*?[:,]",
            r"^For your campaign.*?[:,]",
            r"^This content.*?[:,]"
        ]
        
        for pattern in intro_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
        
        # Ensure we don't have any leading whitespace after stripping intros
        response = response.strip()
        
        # Remove explanatory text at the end if present
        outro_patterns = [
            r"That'?s?( just)?( a suggestion| an example)?[.!].*$",
            r"Hope (this|that) helps[.!].*$",
            r"Let me know.*$",
            r"Feel free to.*$"
        ]
        
        for pattern in outro_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
        
        # Final cleanup
        response = response.strip()
        
        return response



def initialize_chatbot(model_name: str, temperature: float, openai_api_key: str, google_api_key: str) -> None:
    """Initialize the chatbot with specified model and parameters."""
    try:
        if model_name.startswith('gpt'):
            from langchain.chat_models import ChatOpenAI
            st.session_state.conversation = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
        else:
            from langchain.chat_models import ChatGoogleGenerativeAI
            st.session_state.conversation = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=google_api_key
            )
        # Initialize messages, chat_messages, and conversation history in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        st.session_state.conversation = None

def clean_response_format(response: str) -> str:
    """Cleans up the response to ensure natural conversation without JSON formatting."""
    if not isinstance(response, str):
        return str(response)

    # Remove any JSON-like formatting
    response = response.strip()
    if response.startswith('{') or response.startswith('['):
        try:
            import json
            data = json.loads(response)
            if isinstance(data, dict):
                response = data.get('response', '') or data.get('text', '') or data.get('content', '') or str(data)
            elif isinstance(data, list):
                response = '\n'.join(str(item) for item in data if item)
        except:
            # If JSON parsing fails, remove basic JSON markers
            response = re.sub(r'^[{\[]|[}\]]$', '', response)
            response = response.replace('"response":', '').replace('"text":', '').replace('"content":', '')

    # Remove any remaining quotes and clean up
    response = response.replace('"', '').strip()

    # Ensure it starts with a greeting
    greetings = ["Jambo!", "Habari!", "Sawa!", "Karibu!", "Mambo!", "Niaje!", "Hujambo!"]
    if not any(response.startswith(greeting) for greeting in greetings):
        response = f"{random.choice(greetings)} {response}"

    # Add emojis naturally
    emoji_mappings = {
        'campaign': '🌟',
        'success': '✨',
        'marketing': '📢',
        'social media': '📱',
        'email': '📧',
        'content': '📝',
        'sales': '💰',
        'customer': '👥'
    }
    for keyword, emoji in emoji_mappings.items():
        if keyword in response.lower() and emoji not in response:
            response = response.replace(keyword, f"{keyword} {emoji}", 1)

    # Add a natural follow-up if missing
    if not any(char in response[-1] for char in "?!"):
        follow_ups = [
            "\n\nWhat would you like to know more about? 😊",
            "\n\nShall we work on some amazing content together? 🎯",
            "\n\nWhich aspect interests you most? ✨",
            "\n\nWould you like to see some creative ideas? 🌟",
            "\n\nHow can I help you achieve your marketing goals? 💪"
        ]
        response += random.choice(follow_ups)

    return response

def handle_chat_input(user_input: str, model_name: str, temperature: float, openai_api_key: str, google_api_key: str, output_format: Optional[str] = None) -> ChatbotOutput:
    """Handles user input and ensures clean, natural responses without JSON formatting."""
    try:
        if 'conversation' not in st.session_state or not st.session_state.get('conversation'):
            logger.info(f"Initializing chatbot with model: {model_name}")
            initialize_chatbot(model_name, temperature, openai_api_key, google_api_key)

            if not st.session_state.get('conversation') and model_name.startswith('gpt'):
                logger.info("Falling back to Gemini model")
                fallback_model = "gemini-pro"
                st.warning("Pole! OpenAI is taking a break. Switching to our backup system! 🔄")
                initialize_chatbot(fallback_model, temperature, openai_api_key, google_api_key)

            if not st.session_state.get('conversation'):
                error_output = ChatbotOutput.from_text("Connection error. Please check your API keys.")
                st.error(error_output.render())
                return error_output

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append(ChatbotOutput(role="user", content_type="text", text_content=user_input).dict())

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    # Modify the prompt to focus on specific brand content without additional formatting
                    enhanced_prompt = f"""{user_input}

                    IMPORTANT: 
                    - Provide ONLY the requested content/copy for the brand
                    - NO introductions, greetings, or explanations
                    - NO follow-up questions or suggestions
                    - NO emojis, expressions, or casual language
                    - Deliver ONLY the raw content that was requested
                    - Keep formatting minimal and professional"""

                    response = st.session_state.conversation.predict(input=enhanced_prompt)
                    clean_response = clean_response_format(response)
                    full_response += clean_response

                    chatbot_output = None
                    if any(keyword in user_input.lower() for keyword in ["social media", "post", "tweet"]):
                        try:
                            social_media_data = SocialMediaContent.parse_obj(eval(response))
                            chatbot_output = ChatbotOutput(role="assistant", content_type="social_media", social_media_content=social_media_data)
                        except:
                            chatbot_output = ChatbotOutput.from_text(clean_response)
                            logger.warning("Failed to parse response into SocialMediaContent. Using text output.")

                    elif any(keyword in user_input.lower() for keyword in ["email", "mail"]):
                        try:
                            email_data = EmailContent.parse_obj(eval(response))
                            chatbot_output = ChatbotOutput(role="assistant", content_type="email", email_content=email_data)
                        except:
                            chatbot_output = ChatbotOutput.from_text(clean_response)
                            logger.warning("Failed to parse response into EmailContent. Using text output.")

                    elif any(keyword in user_input.lower() for keyword in ["campaign", "marketing", "content"]):
                        try:
                            if isinstance(response, str) and response.startswith('{'):
                                marketing_data = MarketingContent(**eval(response))
                                chatbot_output = ChatbotOutput(role="assistant", content_type="marketing", marketing_content=marketing_data)
                            else:
                                chatbot_output = ChatbotOutput.from_text(clean_response)
                                logger.warning("Expected dictionary for MarketingContent but got a string. Using text output.")
                        except:
                            chatbot_output = ChatbotOutput.from_text(clean_response)
                            logger.warning("Failed to parse response into MarketingContent. Using text output.")
                    else:
                        chatbot_output = ChatbotOutput.from_text(clean_response)

                    # Modify the render_text method in the ChatbotOutput class to remove suggested questions
                    class ModifiedOutput(ChatbotOutput):
                        def render_text(self) -> str:
                            """Renders only the content without extras."""
                            if self.text_content:
                                return self.text_content
                            return ""
                    
                    if isinstance(chatbot_output, ChatbotOutput):
                        modified_output = ModifiedOutput(
                            role=chatbot_output.role,
                            # content_type=chatbot_output.content_type,
                            text_content=chatbot_output.text_content,
                            social_media_content=chatbot_output.social_media_content,
                            email_content=chatbot_output.email_content,
                            marketing_content=chatbot_output.marketing_content
                        )
                        message_placeholder.markdown(modified_output.render_text())
                    else:
                        message_placeholder.markdown(clean_response)
                        
                    st.session_state.messages.append(chatbot_output.dict())
                    return chatbot_output

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    logger.error(error_message)
                    error_output = ChatbotOutput.from_text(error_message)
                    message_placeholder.error(error_output.render())
                    return error_output

    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(error_message)
        return ChatbotOutput.from_text(error_message)


