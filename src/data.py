# data.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Define multiple content types for different marketing needs
class FacebookContent(BaseModel):
    post1: Optional[str] = Field(None, description="First Facebook post")
    post2: Optional[str] = Field(None, description="Second Facebook post")
    imageSuggestion: Optional[str] = Field(None, description="Image suggestion for Facebook posts")

class TwitterContent(BaseModel):
    tweet1: Optional[str] = Field(None, description="First tweet")
    tweet2: Optional[str] = Field(None, description="Second tweet")
    imageSuggestion: Optional[str] = Field(None, description="Image suggestion for tweets")

class InstagramPost(BaseModel):
    caption: Optional[str] = Field(None, description="Instagram post caption")
    imageSuggestion: Optional[str] = Field(None, description="Image suggestion for post")

class InstagramReel(BaseModel):
    reelCaption: Optional[str] = Field(None, description="Instagram reel caption")
    reelSuggestion: Optional[str] = Field(None, description="Suggestion for reel content")

class InstagramContent(BaseModel):
    post1: Optional[InstagramPost] = Field(None, description="First Instagram post")
    post2: Optional[InstagramReel] = Field(None, description="Second Instagram post (Reel)")
    storySuggestion: Optional[str] = Field(None, description="Suggestion for Instagram story")

class SocialMediaContent(BaseModel):
    facebook: Optional[FacebookContent] = Field(None, description="Facebook content")
    twitter: Optional[TwitterContent] = Field(None, description="Twitter content")
    instagram: Optional[InstagramContent] = Field(None, description="Instagram content")

class EmailContent(BaseModel):
    subject_line: str = Field(description="Email subject line")
    preview_text: str = Field(description="Email preview text")
    body: str = Field(description="Main email body")
    call_to_action: str = Field(description="Call to action button text")
    key_benefits: List[str] = Field(description="Key benefits of the product")
    target_market: str = Field(default="General Kenyan households and cooking enthusiasts", description="Target market segment for the campaign")
    campaign_start_date: Optional[datetime] = Field(default_factory=lambda: datetime.now(), description="Campaign start date")
    campaign_end_date: Optional[datetime] = Field(default_factory=lambda: datetime.now() + timedelta(days=30), description="Campaign end date")
    chat_context: Optional[Dict[str, str]] = Field(default_factory=dict, description="Context from previous chat interactions")
    personalization: Optional[Dict[str, str]] = Field(default_factory=dict, description="Personalization based on chat history")
    user_interests: Optional[List[str]] = Field(default_factory=list, description="User interests identified from chat")

@dataclass
class MarketingContent:
    headline: str
    body: str
    call_to_action: str
    key_benefits: List[str]
    target_market: str = field(default="General Kenyan households and cooking enthusiasts")
    campaign_start_date: Optional[datetime] = field(default_factory=lambda: datetime.now())
    campaign_end_date: Optional[datetime] = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    chat_context: Dict[str, str] = field(default_factory=dict)
    conversation_tone: str = field(default="neutral")
    user_feedback: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.campaign_start_date and self.campaign_end_date:
            if self.campaign_end_date <= self.campaign_start_date:
                raise ValueError("End date must be after start date")
        if self.campaign_start_date and self.campaign_start_date < datetime.now():
            raise ValueError("Campaign start date cannot be in the past")

    def to_dict(self):
        return self.__dict__

    def update_from_chat(self, chat_context: Dict[str, str], tone: str, feedback: List[str]):
        self.chat_context.update(chat_context)
        self.conversation_tone = tone
        self.user_feedback.extend(feedback)

# Brand options remain unchanged
BRAND_OPTIONS = {
    "Fresh Fri": "A leading cooking oil brand that provides freshness and quality, enhancing every meal",
    "Salit": "A cooking oil brand that offers great taste and quality, trusted by many Kenyan households",
    "Popco": "A cooking oil brand known for its excellent performance and affordable pricing",
    "Diria": "A premium cooking oil brand known for its high quality and versatility in cooking",
    "Fryking": "A premium cooking oil brand designed for professional and home cooking excellence",
    "Mpishi Poa": "A cooking oil brand that offers superior quality at an affordable price, perfect for everyday use",
    "Pwani SBF": "Specially formulated for high-performance frying with longer-lasting oil quality",
    "Onja": "A trusted brand offering quality oils with a focus on taste and performance",
    "Fresco": "A versatile brand known for quality cooking products and personal care items",
    "Criso": "A cooking oil brand that ensures purity and health with every meal",
    "Tiku": "A reliable cooking oil brand that provides purity and exceptional cooking results",
    "Twiga": "A cooking oil brand known for its great value and high quality for everyday cooking needs",
    "Fresh Zait": "A premium cooking oil made from high-quality ingredients, perfect for healthier cooking",
    "Ndume": "A cooking oil brand that combines quality and affordability for everyday use",
    "Detrex": "A personal care brand specializing in hygiene products with a focus on quality",
    "Frymate": "A trusted cooking oil brand ideal for frying, delivering great taste and performance",
    "Sawa": "A trusted personal care brand offering a variety of soaps for hygiene and skincare",
    "Diva": "A premium personal care brand delivering luxury and effectiveness in every product",
    "Ushindi": "A reliable personal care brand known for quality and affordability",
    "Super Chef": "A popular cooking oil brand used by professional chefs for exceptional frying results",
    "White Wash": "A home care brand offering effective cleaning solutions with outstanding performance",
    "Belleza": "A personal care brand offering luxurious skincare products for a refined experience",
    "Afrisense": "A personal care brand providing a wide range of deodorants and fragrances",
    "Diva": "A personal care brand offering beauty and grooming products for a sophisticated lifestyle",
    "Ushindi": "A brand providing quality hygiene products that cater to everyday needs and ensure freshness"
}