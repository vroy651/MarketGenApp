# prompts.py
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from data import SocialMediaContent, EmailContent, MarketingContent
from rag import RAGSystem  # Assuming rag.py is in the same directory
from llm import get_llm    # Assuming llm.py is in the same directory

from typing import Optional

# Initialize RAG system with a default LLM (Consider moving this to main.py)
# default_llm = get_llm(os.getenv('OPENAI_API_KEY'), "gpt-4", temperature=0)  # Removed, will be passed
# rag_context = RAGSystem(default_llm) # Removed, RAG will be initialized in main


def create_prompt_template(
    instruction: str = """You are a helpful marketing assistant chatbot. The user has provided details about a new marketing campaign.  
    Summarize these details in a friendly and conversational way, as if you are introducing the campaign to the user. 
    Mention the key aspects, but keep it concise. Be enthusiastic! Use Kenyan language and expressions where appropriate 
    to make it feel authentic.
    
    VERY IMPORTANT: DO NOT output JSON format. DO NOT use curly braces like {content}. Your response should be in plain text.
    For example, say "Hello! We're excited about your Fresh Fri campaign!" instead of {"summary": "Hello! We're excited..."}
    
    Always respond in a natural, conversational way. Use phrases like "Jambo!", "Habari!", and other Kenyan expressions to make
    the conversation feel more authentic and relatable. Keep your responses friendly and engaging, focusing on clear communication
    rather than technical formatting.""",
    output_format: str = "Text",
    rag_context_str: Optional[str] = None,
    use_search_engine: bool = False,
    search_engine_prompt_template: Optional[str] = None,
    search_results: Optional[str] = None
):
    """
    Creates a prompt template with modular sections and proper optional field handling.
    Supports JSON output for Pydantic models and plain text output.

    Args:
        instruction (str): The main instruction for content generation.
        output_format (str): Desired output format ("Social Media", "Email", "Marketing", or "Text").
        rag_context_str (Optional[str]):  RAG context string.  If None, RAG is not used.
        use_search_engine (bool):  Whether to include a section for web search results.
        search_engine_prompt_template (Optional[str]): The prompt to use for web search (if enabled).

    Returns:
        PromptTemplate: A LangChain PromptTemplate object.
    """

    # Base Template
    base_template = """
{instruction} for {brand} based on the following details:

**Campaign Details:**
- **Campaign Name:** {campaign_name}
- **Brand:** {brand}
- **SKU:** {sku}
- **Product Category:** {product_category}
"""

    # Optional Campaign Details
    campaign_details = """
**Additional Campaign Information:**
- **Promotion Reference Link:** {promotion_link}
- **Previous Campaign Reference:** {previous_campaign_reference}
- **Campaign Date Range:** {campaign_date_range}
"""

    # Target Market Details
    market_details = """
**Target Market Details:**
- **Age Range:** {age_range}
- **Gender:** {gender}
- **Income Level:** {income_level}
- **Region:** {region}
- **Urban/Rural:** {urban_rural}
"""

    # Specific Instructions
    instructions_section = """
# Handle missing specific instructions
    **Specific Features/Instructions:**
    {specific_instructions}
    Create a catchy, short, and impactful marketing campaign for [Product Name].
The content should highlight the following:

Quality: Emphasize the product’s high standards, superior craftsmanship, or rigorous testing.
Sustainability: Showcase eco-friendly practices, sustainable sourcing, or environmental impact.
Innovation: Focus on the cutting-edge technology, unique features, or benefits the product offers.
Health/Well-being: If relevant, mention how the product aligns with a healthier lifestyle.
Use clear, relatable language that resonates with families, professionals, or target audience groups. Integrate current trends, relatable emotions, or even memes to make the content feel fresh and engaging. Keep the tone warm, inspiring, and authentic.

End with a strong call-to-action encouraging consumers to choose [Product Name] for their everyday needs.

Hashtags: Use trending and relevant hashtags to increase reach.
    """
    # Content Style and Tone
    content_style_section = """
**Content Specifications:**
- **Output Format:** {output_format}
- **Tone and Style:** {tone_style}
"""

    # Combine template sections
    full_template = (
        base_template + campaign_details + market_details + instructions_section + content_style_section
    )

    # Add RAG context section with improved formatting and usage guidelines
    if rag_context_str:
        full_template += """
**Knowledge Base Context:**
{rag_context_str}

**Guidelines for Using Knowledge Base Context:**
1. Ensure accuracy in product details and specifications.
2. Maintain brand voice and messaging consistency.
3. Incorporate relevant historical campaign insights.
4. Reference successful marketing approaches.
5. Align content with company values and guidelines.
"""

    # Conditionally add search results with improved integration
    if use_search_engine and search_results:
        search_results_section = """
**Web Search Results:**
{search_results}

Guidelines for Using Web Search Results:
1. Incorporate current market trends and competitor insights
2. Reference recent industry developments
3. Align content with current consumer sentiments
4. Use data-driven insights to support marketing claims
5. Ensure information is up-to-date and relevant
6. Cross-reference with RAG context when available
7. Prioritize recent and relevant information
"""
        full_template += search_results_section

    # --- Handle different output formats ---
    if output_format in ["Social Media", "Email", "Marketing"]:
        parser_map = {
            "Social Media": PydanticOutputParser(pydantic_object=SocialMediaContent),
            "Email": PydanticOutputParser(pydantic_object=EmailContent),
            "Marketing": PydanticOutputParser(pydantic_object=MarketingContent),
        }
        parser = parser_map[output_format]

        json_instruction = """
**CRITICAL JSON FORMATTING REQUIREMENTS:**
1. Output MUST be a valid JSON object.
2. All property names MUST be in double quotes.
3. String values MUST use double quotes.
4. Apostrophes within text MUST be escaped (e.g., "Pwani Oil\\'s" not "Pwani Oil's").
5. There MUST be NO trailing commas.
6. There MUST be NO additional text or formatting outside the JSON object.
7. The JSON object MUST EXACTLY match this schema:
8. Please only give market content for the given keep it simple.
{format_instructions}

examples: Fresh Fri 500L is on SALE! 🎉 We're talking that same amazing quality you love – golden, delicious, and makes everything taste chef's kiss 🤌. Fresh Fri is all about good food, made with good stuff. We keep it top-notch, and we even care about the planet (because, duh!). Plus, it's the healthy choice for your family's happy tummies.
don't put every time this {Okay, so you want a super catchy campaign for Fresh Fri cooking oil, targeting middle-income, urban women (25-72) in Nairobi and Mombasa, right? Here's a shot, focusing on that "Sale" aspect, and keeping it real:}

Begin JSON output:
"""  # Added "Begin JSON output:"
        full_template += json_instruction
        instruction += "Remember, provide ONLY the final marketing content without any introduction or context. The user wants just the content they can copy and use. "
      
        full_template+=instruction
        input_variables = [
        
                "brand",
                "campaign_name",
                "sku",
                "product_category",
                "promotion_link",
                "previous_campaign_reference",
                "campaign_date_range",
                "age_range",
                "gender",
                "income_level",
                "region",
                "urban_rural",
                "specific_instructions",
                "output_format",
                "tone_style",
                "search_results",
                "rag_context_str" # Include rag_context_str
            ]

        # Add format instructions and instruction as partial variables.
        partial_variables = {
            "format_instructions": parser.get_format_instructions(),
            "instruction": instruction + " Output in strict JSON format.  Do not include any introductory text, labels, or explanations.  Only output the JSON object.", #  More specific instructions
        }


        prompt_template = PromptTemplate(
            template=full_template,
            input_variables=input_variables,
            partial_variables=partial_variables,
            output_parser=parser  # VERY IMPORTANT: Pass the parser here!
        )
        return prompt_template


    else: # Handle Text Output (Non-JSON)
        input_variables=[
                "brand",
                "campaign_name",
                "sku",
                "product_category",
                "promotion_link",
                "previous_campaign_reference",
                "campaign_date_range",
                "age_range",
                "gender",
                "income_level",
                "region",
                "urban_rural",
                "specific_instructions",
                "output_format",
                "tone_style",
                "search_results",
                "rag_context_str" # Include rag_context_str
            ]

        prompt_template = PromptTemplate(
            template=full_template,
            input_variables=input_variables,
            partial_variables={"instruction": instruction},
        )
        return prompt_template
