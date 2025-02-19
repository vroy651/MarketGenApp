import streamlit as st
import time
from config import configure_streamlit_page, load_api_keys, load_css
from data import BRAND_OPTIONS, SocialMediaContent, EmailContent, MarketingContent
from prompt import create_prompt_template
from llm import get_llm
from workflow import create_langraph_workflow
from utils import validate_inputs, save_content_to_file, load_campaign_template, validate_date_range
from rag import RAGSystem
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, ValidationError
# Removed unnecessary imports: No longer directly using ChatbotOutput from chat_handler
from src.chat_history import ChatHistory
import random  # Added for typing simulation
from faker import Faker  # For generating realistic-looking example data

# Configure basic logging (this is fine as is)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

def setup_logging():
    """Set up logging configuration"""
    pass  # No action needed here, basicConfig already set up

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize chat history
chat_history = ChatHistory()

# --- FAKER (for realistic examples) ---
fake = Faker()  # Initialize Faker

class ChatbotOutput(BaseModel):
    """
    Pydantic model to structure the chatbot's output.
    """
    role: str = Field("assistant", description="The role of the message sender (user or assistant)")
    content_type: str = Field("text", description="Type of content (text, social_media, email, marketing, summary)")
    text_content: Optional[str] = Field(None, description="General text content, for simple responses")
    social_media_content: Optional[SocialMediaContent] = Field(None, description="Content for social media posts")
    email_content: Optional[EmailContent] = Field(None, description="Content for email campaigns")
    marketing_content: Optional[MarketingContent] = Field(None, description="General marketing content")
    campaign_summary: Optional[str] = Field(None, description="Summary of the campaign details")
    suggested_actions: Optional[list] = Field(None, description="Suggested actions for the user")
    campaign_details: Optional[dict] = Field(None, description="Summarized campaign details, if applicable")
    alternative_response: Optional[str] = Field(None, description="An alternative response if context is available")

    class Config:
        extra = "allow"

    @classmethod
    def from_text(cls, text: str, role: str = "assistant") -> "ChatbotOutput":
        logger.info(f"Creating text output with role: {role}")
        return cls(role=role, content_type="text", text_content=text)

    @classmethod
    def from_summary(cls, summary: str, role: str = "assistant") -> "ChatbotOutput":
        logger.info(f"Creating summary output with role: {role}")
        return cls(role=role, content_type="summary", campaign_summary=summary)

    def convert_json_to_natural_language(self, response):
        """
        Converts JSON-like responses to natural language.
        """
        logger.debug(f"Converting response to natural language: {response[:100]}...")
        if not isinstance(response, str):
            logger.debug("Response is not a string, converting to string")
            return str(response)

        response = response.strip()
        if (response.startswith('{') and response.endswith('}')) or (response.startswith('[') and response.endswith(']')):
            try:
                logger.debug("Attempting to parse JSON-like response")
                json_content = eval(response)  # Safe in this controlled context

                if isinstance(json_content, dict):
                    keys_to_try = ["campaign_summary", "summary", "text_content", "text", "output"]
                    for key in keys_to_try:
                        if key in json_content:
                            logger.debug(f"Found content in key: {key}")
                            return json_content[key]
                    logger.debug("No specific key found, joining all values")
                    return " ".join(str(v) for v in json_content.values() if v)  # Fallback

                elif isinstance(json_content, list):
                    logger.debug("Converting list content to string")
                    return "\n".join(str(item) for item in json_content if item)

            except (SyntaxError, ValueError, NameError) as e:
                logger.warning(f"Failed to parse JSON-like response: {str(e)}")
                return response  # Not valid JSON-like, return as is

        return response

    def render_text(self) -> str:
        """Renders only the text content, handling both text_content and campaign_summary."""
        logger.debug("Rendering text content")
        if self.content_type == "text" and self.text_content:
            logger.debug("Rendering text_content")
            return self.convert_json_to_natural_language(self.text_content)
        elif self.content_type == "summary" and self.campaign_summary:
            logger.debug("Rendering campaign_summary")
            return self.convert_json_to_natural_language(self.campaign_summary)
        else:
            logger.warning(f"No renderable content found for content_type: {self.content_type}")
            return ""

    def render(self) -> str:
        """Renders the output content to a string based on the content type."""
        if self.content_type == "text":
            return self.render_text()  # Use render_text for text content
        elif self.content_type == "social_media":
            return self.render_social_media()
        elif self.content_type == "email":
            return self.render_email()
        elif self.content_type == "marketing":
            return self.render_marketing()
        elif self.content_type == "summary":
             return self.render_text() # Use render_text for summary
        else:
            return "Unsupported content type."

    def render_social_media(self) -> str:
        """Renders social media content in a simplified format"""
        if not self.social_media_content:
            return "No social media content available."

        content = self.social_media_content
        output = []

        # Use hasattr and getattr for safer access
        if hasattr(content, 'facebook'):
            output.append("\n📘 Facebook")
            fb = content.facebook
            if hasattr(fb, 'post1') and fb.post1:
                output.append(f"Post 1:\n{fb.post1}")
            if hasattr(fb, 'post2') and fb.post2:
                output.append(f"\nPost 2:\n{fb.post2}")
            if hasattr(fb, 'imageSuggestion') and fb.imageSuggestion:
                output.append(f"\nImage Suggestion:\n{fb.imageSuggestion}")

        if hasattr(content, 'twitter'):
            output.append("\n\n🐦 Twitter")
            tw = content.twitter
            if hasattr(tw, 'tweet1') and tw.tweet1:
                output.append(f"Tweet 1:\n{tw.tweet1}")
            if hasattr(tw, 'tweet2') and tw.tweet2:
                output.append(f"\nTweet 2:\n{tw.tweet2}")
            if hasattr(tw, 'imageSuggestion') and tw.imageSuggestion:
                output.append(f"\nImage Suggestion:\n{tw.imageSuggestion}")


        if hasattr(content, 'instagram'):
            output.append("\n\n📸 Instagram")
            ig = content.instagram
            if hasattr(ig, 'post1') and ig.post1:
                output.append(f"Post 1:\n{ig.post1.get('caption', '')}")
                if 'imageSuggestion' in ig.post1:
                    output.append(f"\nImage Suggestion:\n{ig.post1['imageSuggestion']}")
            if hasattr(ig, 'post2') and ig.post2:
                output.append(f"\nPost 2 (Reel):\n{ig.post2.get('reelCaption', '')}")
                if 'reelSuggestion' in ig.post2:
                    output.append(f"\nReel Suggestion:\n{ig.post2['reelSuggestion']}")
            if hasattr(ig, 'storySuggestion') and ig.storySuggestion:
                output.append(f"\nStory Suggestion:\n{ig.storySuggestion}")

        return "\n".join(output)

    def render_email(self) -> str:
        if not self.email_content:
            return "No email content available."

        content = self.email_content
        output = [
            "**Email Campaign: [Campaign Name/Description]**\n",
            f"*   **Subject Line:** {content.subject_line}",
            f"*   **Preview Text:** {content.preview_text}",
            f"*   **Body:**\n\n    > {content.body.replace('  ', ' ').replace('\n', '\n> ')}",
            f"*   **Call to Action:** {content.call_to_action}",
            "*   **Key Benefits:**",
            *[f"    *   {benefit}" for benefit in content.key_benefits],
            f"*   **Target Market:** {content.target_market}",
            f"*   **Campaign Dates:** {content.campaign_start_date.strftime('%Y-%m-%d') if content.campaign_start_date else 'Not set'} to {content.campaign_end_date.strftime('%Y-%m-%d') if content.campaign_end_date else 'Not set'}",
        ]
        return "\n".join(output)

    def render_marketing(self) -> str:
        if not self.marketing_content:
            return "No marketing content available."

        content = self.marketing_content
        output = [
           "**Marketing Content: [Campaign Name/Description]**\n",
            f"*   **Headline:** {content.headline}",
            f"*   **Body:**\n\n    > {content.body.replace('  ', ' ').replace('\n', '\n> ')}",
            f"*   **Call to Action:** {content.call_to_action}",
            "*   **Key Benefits:**",
            *[f"    *   {benefit}" for benefit in content.key_benefits],
            f"*   **Target Market:** {content.target_market}",
            f"*   **Campaign Dates:** {content.campaign_start_date.strftime('%Y-%m-%d') if content.campaign_start_date else 'Not set'} to {content.campaign_end_date.strftime('%Y-%m-%d') if content.campaign_end_date else 'Not set'}",
        ]
        return "\n".join(output)

def clear_chat_history():
    """Clears the chat history, both in Streamlit and the external ChatHistory."""
    st.session_state.messages = []  # Clear Streamlit's messages
    if "conversation" in st.session_state and hasattr(st.session_state.conversation, "memory"):
        st.session_state.conversation.memory.clear()  # Clear Langchain memory
    chat_history.clear_history()  # Clear the external chat history

def delete_message(index: int):
    """Deletes a specific message from the session state."""
    if "messages" in st.session_state and 0 <= index < len(st.session_state.messages):
        # Remove from Streamlit session state
        del st.session_state.messages[index]

        # Remove from Langchain memory, if it exists and is accessible
        if "conversation" in st.session_state and hasattr(st.session_state.conversation, "memory"):
            memory = st.session_state.conversation.memory
            if hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
                if 0 <= index < len(memory.chat_memory.messages):
                    del memory.chat_memory.messages[index]

        # Remove from external chat history
        chat_history.delete_message(index)


def initialize_rag_system(openai_api_key):
    """Initialize RAG system."""
    try:
        if 'rag_system' not in st.session_state:
            logger.info("Initializing new RAG system")
            llm = get_llm(openai_api_key, "gpt-4-turbo-preview", temperature=0)  # gpt-4 is deprecated, using gpt-4-turbo
            if not llm:
                logger.error("Failed to initialize LLM")
                raise ValueError("Failed to initialize LLM")

            st.session_state.rag_system = RAGSystem(llm)
            logger.info("RAG system instance created successfully")

            if not hasattr(st.session_state.rag_system, 'vector_store') or st.session_state.rag_system.vector_store is None:
                st.info("🔄 Loading knowledge base...")
                try:
                    logger.info("Attempting to load documents from knowledge base")
                    loader = TextLoader("../cleaned_cleaned_output.txt")
                    documents = loader.load()
                    logger.info(f"Loaded {len(documents)} documents from knowledge base")

                    if st.session_state.rag_system.ingest_documents(documents):
                        st.success("✨ RAG system initialized successfully")
                        logger.info("Documents ingested successfully into RAG system")
                    else:
                        error_msg = "Failed to ingest documents into RAG system"
                        logger.warning(error_msg)
                        st.warning(error_msg + " - will proceed without context")
                except Exception as doc_error:
                    error_msg = f"Error loading documents: {str(doc_error)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    raise
            else:
                st.info("✨ Using existing RAG knowledge base")
                logger.info("Using cached RAG knowledge base")
        else:
            logger.info("Using existing RAG system from session state")

    except Exception as e:
        error_msg = f"RAG system initialization failed: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg + " - will proceed without context")
        if 'rag_system' in st.session_state:
            del st.session_state.rag_system


def apply_template_defaults(template_type):
    """Apply default values from a campaign template, using Faker for realistic examples."""
    if template_type != "Custom Campaign":
        template_data = load_campaign_template(template_type)
        if template_data:
            for key, value in template_data.items():
                # Use Faker for realistic defaults where appropriate
                if key == "campaign_name":
                    st.session_state[key] = fake.bs().title()  # Catchy campaign name
                elif key == "promotion_link":
                    st.session_state[key] = fake.url()
                elif key == "campaign_date_range":
                    start_date = fake.date_this_year(before_today=False, after_today=True)
                    end_date = fake.date_between_dates(date_start=start_date, date_end="+30d")
                    st.session_state[key] = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                elif key == "specific_instructions":
                    st.session_state[key] = fake.paragraph(nb_sentences=3)
                else:
                    st.session_state[key] = value  # Use template value if no Faker replacement
        else:
            st.warning(f"Template '{template_type}' not found.")
    else:
        #Clear defaults when "Custom Campaign is selected"
        st.session_state.campaign_name = ""
        st.session_state.promotion_link = ""
        st.session_state.previous_campaign_reference = ""
        st.session_state.campaign_date_range = ""
        st.session_state.specific_instructions = ""


def generate_initial_context(
    input_vars, model_name, temperature, top_p, use_rag, rag_query,
    use_search_engine, search_engine_query, selected_brand, openai_api_key, google_api_key, output_format
):
    """
    Generates an initial context message (campaign summary).
    """
    llm = get_llm(google_api_key if not model_name.startswith("gpt") else openai_api_key, model_name, temperature, top_p)
    if not llm:
        return ChatbotOutput.from_text("Error: Failed to initialize LLM.")

    prompt_template = create_prompt_template(
        instruction="""You are a helpful marketing assistant chatbot. The user has provided details about a new marketing campaign.  
        Summarize these details in a friendly and conversational way, as if you are introducing the campaign to the user. 
        Mention the key aspects, but keep it concise. Be enthusiastic! Use Kenyan language and expressions where appropriate 
        to make it feel authentic.
        
        VERY IMPORTANT: Your response MUST be in natural conversational language. DO NOT output JSON or structured data.
        Use friendly Kenyan expressions like 'Jambo!', 'Karibu!', or 'Habari!' to make the conversation warm and engaging.
        Focus on creating a welcoming atmosphere while discussing the campaign details.""",
        output_format="text",  # Force text output
        use_search_engine=use_search_engine,
        search_engine_prompt_template=search_engine_query
    )

    try:
        if use_rag and rag_query:
            context_query = f"""
            Brand: {selected_brand}
            Product: {input_vars.get('sku', 'N/A')}
            Category: {input_vars.get('product_category', 'N/A')}
            Query: {rag_query}
            """
            rag_context = st.session_state.rag_system.query(context_query)
            if rag_context:
                input_vars["rag_context"] = rag_context

        workflow = create_langraph_workflow(
            llm,
            prompt_template,
            input_vars,
            "text",
            use_search_engine=use_search_engine,
            search_engine_query=search_engine_query if use_search_engine else None
        )

        result = workflow.invoke(input_vars)

        if "error" in result:
            return ChatbotOutput.from_text(f"Error generating context: {result['error']}")

        generated_context = result.get("output", "")
        summary_text = ChatbotOutput.from_text(generated_context).render_text() # Use render_text
        return ChatbotOutput.from_summary(summary_text)

    except Exception as e:
        return ChatbotOutput.from_text(f"Error generating context: {str(e)}")



def initialize_chatbot(model_name, temperature, openai_api_key, google_api_key, initial_context=""):
    """Initializes the chatbot and ensures messages are initialized."""
    llm = get_llm(google_api_key, model_name, temperature)
    if not llm and model_name.startswith("gpt"):
        llm = get_llm(openai_api_key, model_name, temperature)

    if 'conversation' not in st.session_state:
        memory = ConversationBufferMemory()
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory
        )

    # Initialize messages list (always initialize or ensure it exists)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add initial context if provided and messages list is empty
    if initial_context and not st.session_state.messages:
        st.session_state.messages.append(initial_context.dict())
        # Add to Langchain memory as well
        st.session_state.conversation.memory.save_context({"input": "Hello!"}, {"output": initial_context.render_text()})



def handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key, output_format):
    """Handles user input, generates responses, and simulates typing."""
    try:
        context_updates = chat_history.get_full_context()

        if 'conversation' not in st.session_state or not st.session_state.get('conversation'):
            logger.info(f"Initializing chatbot with model: {model_name}")
            initialize_chatbot(model_name, temperature, openai_api_key, google_api_key)

            if not st.session_state.get('conversation') and model_name.startswith('gpt'):
                logger.info("Falling back to Gemini model")
                fallback_model = "gemini-pro"
                st.warning(f"OpenAI authentication failed. Falling back to {fallback_model}...")
                initialize_chatbot(fallback_model, temperature, openai_api_key, google_api_key)

            if not st.session_state.get('conversation'):
                error_output = ChatbotOutput.from_text("Failed to initialize chatbot. Please verify your API keys in the .env file.")
                st.error(error_output.render())
                return error_output

        chat_history.add_message("user", user_input, metadata={"context": context_updates})

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append(ChatbotOutput(role="user", content_type="text", text_content=user_input).dict())

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):  #  spinner
                try:
                    chatbot_output = None
                    modified_input = f"""{user_input}

                    VERY IMPORTANT: DO NOT output JSON format. DO NOT use curly braces like {{content}}. 
                    Your response should be in plain text.  Use Kenyan expressions like 'Jambo!', 'Karibu!', or 'Habari!' 
                    to make the conversation warm and engaging."""

                    context_aware_input = f"{modified_input}\n\nContext Updates: {context_updates}"
                    response = st.session_state.conversation.predict(input=context_aware_input)

                    chat_history.add_message("assistant", response, metadata={"context": context_updates})

                    # Attempt to parse, but handle potential errors gracefully
                    import json
                    try:
                        if isinstance(response, str):
                            try:
                                response_dict = json.loads(response)
                            except json.JSONDecodeError:
                                response_dict = response
                        else:
                            response_dict = response

                        if isinstance(response_dict, dict):
                            if any(keyword in user_input.lower() for keyword in ["social media", "post", "tweet"]):
                                social_media_data = SocialMediaContent(**response_dict)
                                chatbot_output = ChatbotOutput(role="assistant", content_type="social_media", social_media_content=social_media_data)
                            elif any(keyword in user_input.lower() for keyword in ["email", "mail"]):
                                email_data = EmailContent(**response_dict)
                                chatbot_output = ChatbotOutput(role="assistant", content_type="email", email_content=email_data)
                            elif any(keyword in user_input.lower() for keyword in ["campaign", "marketing", "content"]):
                                marketing_data = MarketingContent(**response_dict)
                                chatbot_output = ChatbotOutput(role="assistant", content_type="marketing", marketing_content=marketing_data)
                            else:
                                chatbot_output = ChatbotOutput.from_text(response)
                        else:
                            chatbot_output = ChatbotOutput.from_text(response)
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.debug(f"Content type parsing failed: {str(e)}")
                        chatbot_output = ChatbotOutput.from_text(response)
                    except Exception as e:
                        logger.exception(f"Unexpected error during parsing: {e}")
                        chatbot_output = ChatbotOutput.from_text(response)
                        st.error("An unexpected error occurred during response processing.")


                    # --- TYPING SIMULATION ---
                    full_response = chatbot_output.render()  # Get full response
                    displayed_response = ""
                    for char in full_response:
                        displayed_response += char
                        time.sleep(random.uniform(0.02, 0.06))  # Simulate typing speed
                        message_placeholder.markdown(displayed_response + "▌")
                    message_placeholder.markdown(displayed_response) # Remove cursor

                    if any(keyword in user_input.lower() for keyword in ["campaign", "marketing", "content"]):
                        campaign_name = st.session_state.get("campaign_name", "Unnamed Campaign")  # Use .get()
                        saved_file = save_content_to_file(chatbot_output.render(), campaign_name, "txt")
                        if saved_file:
                            st.info(f"💾 Content saved to: {saved_file}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating response: {error_msg}")

                    if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                        if model_name.startswith('gpt'):
                            fallback_model = "gemini-pro"
                            st.warning(f"OpenAI authentication failed. Attempting to use {fallback_model}...")
                            initialize_chatbot(fallback_model, temperature, openai_api_key, google_api_key)
                            if st.session_state.get('conversation'):
                                response = st.session_state.conversation.predict(input=user_input)
                                chatbot_output = ChatbotOutput.from_text(response)
                                full_response += chatbot_output.render_text()  # render_text
                                message_placeholder.markdown(chatbot_output.render_text() + "▌")
                            else:
                                error_output = ChatbotOutput.from_text("Failed to fall back to alternative model. Please check your API keys.")
                                st.error(error_output.render())
                                return error_output
                        else:
                            error_output = ChatbotOutput.from_text("Authentication failed. Please check your API keys in the .env file.")
                            st.error(error_output.render())
                            return error_output
                    else:
                        error_output = ChatbotOutput.from_text(f"Error generating response: {error_msg}")
                        st.error(error_output.render())
                        return error_output
            #message_placeholder.markdown(chatbot_output.render_text()) # Replaced by typing simulation

        st.session_state.messages.append(chatbot_output.dict())
        return chatbot_output


    except Exception as e:
        logger.error(f"Error in chat input handling: {str(e)}")
        error_output = ChatbotOutput.from_text("An unexpected error occurred. Please try again.")
        st.error(error_output.render())
        return error_output

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'gpt-3.5-turbo'

def display_model_selector():
    """Display model selection dropdown"""
    models = [
        "gemini-2.0-pro-exp-02-05"
    ]
    st.sidebar.selectbox(
        'Select AI Model',
        models,
        key='selected_model',
        help='Choose the AI model for generating responses'
    )

def main():
    configure_streamlit_page()
    load_css()
    google_api_key, openai_api_key = load_api_keys()
    initialize_rag_system(openai_api_key)

    # --- UI ENHANCEMENTS ---
    # st.image("path/to/pwani_logo.png", width=200)  # Add Pwani Oil logo (replace path)
    st.title("🌟 Pwani Oil Marketing Assistant Chatbot")
    st.subheader("Your AI-Powered Marketing Campaign Partner")
    st.caption("Provide campaign details below, and then chat with the AI to generate content!")

    with st.sidebar:
        st.header("⚙️ Chatbot Settings")
        st.markdown("**Choose your AI model and settings here.**")

        display_model_selector()

        model_name = st.session_state.selected_model

        temperature = st.slider("🎨 Creativity", 0.0, 1.0, 0.7, key="temperature_slider",
                                help="Higher values: more random/creative output. Lower values: more predictable.")

        st.subheader("🧠 Advanced Options")

        use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True,
                                help="Use Pwani Oil's knowledge base for more accurate responses.")
        use_search_engine = st.checkbox("Use Web Search (Optional)", value=False,
                                        help="Search the web for additional information.")
        if use_search_engine:
            search_engine_query = st.text_input("🔍 Search Query", key="search_query_input",
                                                help="Enter a search query (e.g., 'competitor marketing campaigns').")
        else:
            search_engine_query = None

    with st.expander("📝 Campaign Details", expanded=True):
        st.markdown("**Tell us about your marketing campaign!**")
        template_type = st.selectbox(
            "🚀 Campaign Type",
            ["Custom Campaign", "Product Launch", "Seasonal Sale", "Brand Awareness"],
            key="template_type",
            on_change=apply_template_defaults,
            args=("Custom Campaign",),  # Pass the default value
            help="Choose a template or 'Custom Campaign'."
        )

        col1, col2 = st.columns(2)
        with col1:
            campaign_name = st.text_input("🎯 Campaign Name", key="campaign_name",
                                          placeholder="e.g.,  Safi Fresh Launch",
                                          help="Give your campaign a catchy name!")
            selected_brand = st.selectbox("⭐ Brand", options=list(BRAND_OPTIONS.keys()),
                                        help="Select the brand.")
            if selected_brand:
                st.info(f"📝 Brand Description: {BRAND_OPTIONS[selected_brand]}")

            promotion_link = st.text_input("🔗 Promotion Link (optional)", key="promotion_link",
                                          placeholder="e.g., www.pwani.com/safi-fresh",
                                        help="Website/landing page URL.")
            previous_campaign_reference = st.text_input("⏮️ Previous Campaign Ref (optional)", key="previous_campaign_reference",
                                                        placeholder="e.g., 2023 Holiday Campaign",
                                                        help="Reference to a similar campaign.")
        with col2:
            sku = st.selectbox("📦 SKU", ["500L", "250L", "1L", "10L", "20L", "2L", "3L", "5L", "10KG", "500G", "1KG", "2KG", "17KG", "4KG", "100G", "700G", "800G", "600G", "80G", "125G", "175G", "200G", "225G", "20G"], key="sku",
                            help="Select the Stock Keeping Unit (SKU).")
            product_category = st.selectbox("🏷️ Product Category", ["Cooking Oil", "Cooking Fat", "Bathing Soap", "Home Care", "Lotion", "Margarine", "Medicine Soap"], key="product_category",
                                            help="Select the product category.")
            campaign_date_range = st.text_input("📅 Date Range (YYYY-MM-DD to YYYY-MM-DD)", key="campaign_date_range",
                                                placeholder="e.g., 2024-03-15 to 2024-04-15",
                                                help="Start and end dates (YYYY-MM-DD to YYYY-MM-DD).")
            tone_style = st.selectbox("🎤 Tone & Style", ["Professional", "Casual", "Friendly", "Humorous", "Formal", "Inspirational", "Educational", "Persuasive", "Emotional"], key="tone_style",
                                    help="Overall tone for your marketing content.")

            output_format = st.selectbox("📄 Output Format", ["Text", "Social Media", "Email", "Marketing"],
                                        help="Primary type of content to generate.",
                                        key="output_format")

        st.subheader("👥 Target Audience")
        col1, col2 = st.columns(2)

        with col1:
            age_range = st.select_slider("🎂 Age Range", options=list(range(18, 76, 1)), value=(25, 45), key="age_range_slider")
            gender = st.multiselect("🚻 Gender", ["Male", "Female", "Other"], default=["Female"], key="gender_multiselect")

        with col2:
            income_level = st.select_slider("💰 Income Level", options=["Low", "Middle Low", "Middle", "Middle High", "High"], value="Middle", key="income_level_slider")
            region = st.multiselect("🌍 Region", ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Other"], default=["Nairobi", "Mombasa"], key="region_multiselect")
            urban_rural = st.multiselect("🏘️ Area Type", ["Urban", "Suburban", "Rural"], default=["Urban"], key="urban_rural_multiselect")

        st.subheader("✍️ Additional Instructions")
        specific_instructions = st.text_area("Additional instructions for the AI (e.g., keywords, goals, competitors).", key="specific_instructions_input", placeholder="e.g., Focus on health benefits, compare with competitor X, target young families.")

        if st.button("🚀 Submit", type="primary", use_container_width=True):
            input_vars = {
                "campaign_name": campaign_name,
                "promotion_link": promotion_link,
                "previous_campaign_reference": previous_campaign_reference,
                "sku": sku,
                "product_category": product_category,
                "campaign_date_range": campaign_date_range,
                "age_range": f"{age_range[0]}-{age_range[1]}" if age_range else None,
                "gender": ", ".join(gender) if gender else None,
                "income_level": income_level if income_level else None,
                "region": ", ".join(region) if region else None,
                "urban_rural": ", ".join(urban_rural) if urban_rural else None,
                "specific_instructions": specific_instructions,
                "brand": selected_brand,
                "tone_style": tone_style,
                "search_results": None,
                "template_type": template_type,
                "output_format": output_format
            }

            is_valid, error_message = validate_inputs(input_vars)
            if not is_valid:
                st.error(error_message)  # Display detailed error
                st.stop()
            if not validate_date_range(campaign_date_range):
                st.error("Invalid date range. Please use YYYY-MM-DD to YYYY-MM-DD format.")  # Specific date error
                st.stop()

            with st.spinner("Generating initial context..."):
                initial_context_output = generate_initial_context(
                    input_vars, model_name, temperature, 0.9, use_rag,
                    specific_instructions, use_search_engine, search_engine_query,
                    selected_brand, openai_api_key, google_api_key, output_format
                )

            initialize_chatbot(model_name, temperature, openai_api_key, google_api_key, initial_context_output)

            # Check for initialization and append initial context if needed
            if "messages" not in st.session_state:
                st.session_state.messages = []  # Initialize if not present
            if initial_context_output and not st.session_state.messages:  # Only add if empty
                st.session_state.messages.append(initial_context_output.dict())

            st.success("Chatbot initialized! You can now start chatting.")

    # --- CHAT DISPLAY (Enhanced Styling) ---
    if "messages" in st.session_state:
         # Clear button
        if st.session_state.messages:  # Only show if there are messages
             if st.button("🗑️ Clear All Messages", key="clear_all", use_container_width=True):
                clear_chat_history()
                st.rerun()  # Correctly placed to refresh the UI

         # Display existing messages
        for i, message_data in enumerate(st.session_state.messages):
            try:
                message = ChatbotOutput(**message_data)
            except ValidationError as e:
                logger.error(f"Error loading message {i}: {e}")
                st.error(f"Error loading message {i}. See logs.")
                continue

            col1, col2 = st.columns([0.9, 0.1])  # Adjust for delete button
            with col1:
                with st.chat_message(message.role):
                    st.markdown(message.render())  # Use the general render() method
            with col2:
                if st.button("🗑️", key=f"delete_{i}", help="Delete message"):
                    delete_message(i)
                    st.rerun()  # Force re-render after deletion

        if user_input := st.chat_input("Ask me anything about the campaign..."):
            handle_chat_input(user_input, model_name, temperature, openai_api_key, google_api_key, output_format)

if __name__ == "__main__":
    main()