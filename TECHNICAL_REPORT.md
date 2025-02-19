# Technical Report: Pwani Oil Marketing Content Generator

## Overview
The Pwani Oil Marketing Content Generator is a Streamlit-based application that leverages AI and RAG (Retrieval-Augmented Generation) technology to create targeted marketing content. The application combines modern UI elements with sophisticated natural language processing to generate customized marketing campaigns across multiple channels.

## Architecture

### Core Components

1. **User Interface (ui.py)**
   - Built with Streamlit for interactive web interface
   - Features tabbed navigation for campaign details, target market, and advanced settings
   - Implements real-time content generation and preview
   - Supports dynamic template selection and customization

2. **RAG System (rag.py)**
   - Implements Retrieval-Augmented Generation for context-aware content
   - Uses FAISS for efficient vector storage and retrieval
   - Features hybrid retrieval combining BM25 and vector search
   - Supports GPU acceleration for improved performance
   - Implements caching mechanism for optimized retrieval

3. **Workflow Engine (workflow.py)**
   - Utilizes LangGraph for orchestrating content generation
   - Implements state management for complex generation flows
   - Handles error cases and provides detailed logging
   - Supports both synchronous and asynchronous operations

### Key Features

1. **Content Generation**
   - Multi-format output support (Social Media, Email, Marketing, Text)
   - Template-based generation with customization options
   - Real-time content preview and editing
   - Image generation capabilities for marketing assets

2. **Campaign Management**
   - Customizable campaign templates
   - Target audience specification
   - Campaign date range management
   - Brand and product categorization

3. **Advanced Features**
   - RAG-powered contextual content generation
   - Web search integration for real-time market data
   - Conversation memory for consistent interactions
   - Multi-model support (GPT-4, Gemini Pro)

## Technical Implementation

### RAG System Implementation

```python
class RAGSystem:
    - Utilizes FAISS for vector storage
    - Implements hybrid retrieval (BM25 + Vector)
    - Features contextual compression for improved relevance
    - Supports conversation memory for context retention
```

### Content Generation Workflow

1. **Input Processing**
   - Validates user inputs
   - Applies template defaults
   - Processes target market parameters

2. **Content Generation**
   - Retrieves relevant context from RAG system
   - Generates content using selected LLM
   - Applies format-specific transformations

3. **Output Handling**
   - Formats generated content
   - Handles image generation requests
   - Manages content storage and retrieval

## Configuration

### RAG System Configuration
- Chunk size: 500 tokens
- Chunk overlap: 50 tokens
- Similarity threshold: 0.7
- BM25 weight: 0.3
- Vector weight: 0.7

### Model Configuration
- Supports multiple LLM models:
  - GPT-4
  - Gemini Pro
  - Gemini 1.5 Pro
  - Gemini 2.0 Flash

## Performance Considerations

1. **Optimization Techniques**
   - RAG context caching
   - GPU acceleration for FAISS
   - Chunked document processing
   - Efficient memory management

2. **Error Handling**
   - Graceful degradation for RAG failures
   - Rate limit management
   - Timeout handling
   - Input validation

## Future Improvements

1. **Potential Enhancements**
   - Enhanced template management
   - Advanced analytics integration
   - Multi-language support
   - A/B testing capabilities

2. **Technical Debt**
   - Implement comprehensive testing
   - Enhance error recovery mechanisms
   - Optimize memory usage
   - Improve documentation coverage

## Conclusion
The Pwani Oil Marketing Content Generator demonstrates a sophisticated implementation of AI-powered content generation with RAG capabilities. The modular architecture and robust error handling make it a reliable tool for marketing content creation, while the integration of modern AI models ensures high-quality output.