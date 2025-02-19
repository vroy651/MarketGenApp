# Pwani Oil Marketing Content Generator

An AI-powered content generation application designed specifically for Pwani Oil's marketing campaigns. This tool leverages advanced language models to create engaging, brand-aligned content across multiple marketing channels.

## Features

- **Campaign Management**
  - Create and manage marketing campaigns
  - Customize campaign templates
  - Set target audience specifications
  - Define campaign date ranges

- **AI-Powered Content Generation**
  - Multi-format content support:
    - Social media posts
    - Email campaigns
    - Marketing copy
    - General text content
  - Real-time content preview
  - Template-based generation
  - Context-aware content creation

- **Advanced Capabilities**
  - RAG (Retrieval-Augmented Generation) system
  - Multi-model support (GPT-4, Gemini Pro)
  - Web search integration
  - Conversation memory for consistent interactions

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory
2. Add your API keys:
   ```env
   GOOGLE_API_KEY="your-google-api-key"
   OPENAI_API_KEY="your-openai-api-key"
   ```
   Note: At least one API key (Google or OpenAI) is required

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Access the web interface at `http://localhost:8501`
3. Create a new campaign or select an existing template
4. Input your content requirements
5. Generate and customize content
6. Save or export your content

## System Requirements

- Python 3.10 or higher
- Internet connection for API access
- Sufficient RAM for model operations (4GB minimum recommended)

## Support

For issues and feature requests, please create an issue in the repository.

## License

Proprietary - All rights reserved