from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
import logging
import os
import warnings
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import GPU FAISS (more robust handling)
try:
    import faiss

    GPU_FAISS_AVAILABLE = hasattr(faiss, 'GpuIndexFlatL2')
    if GPU_FAISS_AVAILABLE:
        logger.info("GPU FAISS is available")
    else:
        logger.info("GPU FAISS is not available, using CPU version")
except ImportError:
    GPU_FAISS_AVAILABLE = False
    logger.warning("FAISS import failed, will use CPU-only version")
    warnings.warn("FAISS import failed, will use CPU-only version", ImportWarning)


class RAGSystemConfig(BaseModel):
    """Configuration for the RAG system."""
    chunk_size: int = Field(500, description="Size of text chunks")
    chunk_overlap: int = Field(50, description="Overlap between text chunks")
    similarity_threshold: float = Field(0.7, description="Similarity threshold for embeddings filter")
    bm25_weight: float = Field(0.3, description="Weight for BM25 retriever")
    vector_weight: float = Field(0.7, description="Weight for vector retriever")
    index_path: str = Field("ContentGenApp/faiss_index", description="Path to save/load FAISS index")
    knowledge_base_path: str = Field("cleaned_cleaned_output.txt", description="path to knowledge base")
    use_summary_memory: bool = Field(False, description="Whether to use ConversationSummaryBufferMemory instead of ConversationBufferMemory")
    max_token_limit: int = Field(3000, description="Max token limit for summary memory (if used)")  # Add a max_token_limit
    use_tavily_search: bool = Field(True, description="Whether to use Tavily search instead of DuckDuckGo")


class RAGSystem:
    """Retrieval-Augmented Generation (RAG) System."""

    def __init__(self, llm, embedding_model=None, openai_api_key=None, config: Optional[RAGSystemConfig] = None):
        self.llm = llm
        try:
            self.embedding_model = embedding_model or OpenAIEmbeddings(openai_api_key=openai_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
            # Try to load existing index first
            self._load_existing_index()
            if not self.vector_store:
                raise
        self.config = config or RAGSystemConfig()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.vector_store = None
        self.index_path = self.config.index_path
        self.knowledge_base_path = self.config.knowledge_base_path
        
        # Initialize Tavily search with basic configuration
        try:
            self.tavily_search = TavilySearchAPIWrapper()
            logger.info("Tavily search initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily search: {e}")
            self.tavily_search = None

        if self.config.use_summary_memory:
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=self.config.max_token_limit
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

    def _load_existing_index(self):
        """Load existing FAISS index if available, otherwise initialize empty vector store."""
        try:
            if os.path.exists(os.path.join(self.index_path, "index.faiss")):
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.vector_store = FAISS.load_local(self.index_path, self.embedding_model, allow_dangerous_deserialization=True)
                logger.info("FAISS index loaded successfully")
            else:
                logger.info("No existing FAISS index found")
                self.vector_store = None
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.vector_store = None

    def ingest_documents(self, documents: List[Document]) -> bool:
        """Ingest documents into the vector store.

        Args:
            documents: List of documents to ingest.

        Returns:
            bool: True if ingestion was successful, False otherwise.
        """
        try:
            if not documents:
                logger.warning("No documents provided for ingestion")
                return False

            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            if not texts:
                logger.warning("No text chunks created from documents")
                return False

            # Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new FAISS index")
                self.vector_store = FAISS.from_documents(texts, self.embedding_model)
            else:
                logger.info("Adding documents to existing FAISS index")
                self.vector_store.add_documents(texts)

            # Save the index
            os.makedirs(self.index_path, exist_ok=True)
            self.vector_store.save_local(self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")
            return True

        except Exception as e:
            logger.exception(f"Error ingesting documents: {e}")
            return False

    def _create_retriever(self, documents:List[Document], k: int = 3, use_web_search: bool = False):
        """Creates and configures the retriever chain with optional web search."""

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # Create vector retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        retrievers = [bm25_retriever, vector_retriever]
        weights = [self.config.bm25_weight * 0.8, self.config.vector_weight * 0.8]

        if use_web_search:
            if self.config.use_tavily_search and self.tavily_search:
                # Use Tavily search with enhanced error handling and result processing
                logger.info("Using Tavily search for web retrieval")
                def web_search(q):
                    try:
                        # Enhance search query with context
                        enhanced_query = f"Pwani Oil {q}"
                        logger.info(f"Enhanced Tavily search query: {enhanced_query}")
                        
                        # Get search results
                        results = self.tavily_search.run(enhanced_query)
                        logger.info(f"Tavily search results received: {len(str(results))} chars")
                        
                        # Process and structure the results
                        if isinstance(results, str):
                            return [Document(page_content=results)]
                        elif isinstance(results, dict) and 'text' in results:
                            return [Document(page_content=results['text'])]
                        else:
                            return [Document(page_content=str(results))]
                    except Exception as e:
                        logger.error(f"Tavily search error: {e}")
                        return [Document(page_content="Search failed")]
            else:
                # Fallback to DuckDuckGo
                logger.info("Falling back to DuckDuckGo search")
                web_search = DuckDuckGoSearchRun()
                web_search = lambda q: [
                    Document(page_content=(
                        logger.info(f"DuckDuckGo search query: {q}") or
                        logger.info(f"DuckDuckGo search result: {web_search.run(q)}") or
                        web_search.run(q)
                    ))
                ]
            
            retrievers.append(web_search)
            weights.append(0.4)  # Assign weight to web search results

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )

        # Create contextual compression retriever
        re_ranker = EmbeddingsFilter(embeddings=self.embedding_model, similarity_threshold=self.config.similarity_threshold)
        compressor = LLMChainExtractor.from_llm(self.llm)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        return compression_retriever


    def query(self, question: str, k: int = 3, use_web_search: bool = True):
        """
        Queries the RAG system.

        Args:
            question: The question to ask.
            k: The number of documents to retrieve.
            use_web_search: Whether to include web search results in the retrieval process.

        Returns:
            The answer to the question, or an empty string if an error occurs.
        """
        if not isinstance(question, str) or not question.strip():
            logger.warning("Question is empty or invalid.")
            return ""

        if not self.vector_store:
            logger.warning("No documents have been ingested yet.")
            return ""

        try:
            # get documents from vector store. Need raw documents for BM25
            docs = self.vector_store.similarity_search(question, k=k*3)  # Fetch more, compression will reduce
            if not docs:
                logger.info("No documents found for the question.")
                return ""

            retriever = self._create_retriever(docs, k, use_web_search=use_web_search)

            # --- Create the Conversational Chain ---
            #   Using LCEL for better control and transparency
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer questions based on the provided context and chat history. "
                           "If the answer is not in the context or chat history, say 'I don't know'."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Context: {context}\nQuestion: {input}")
            ])

            # Load chat history with proper error handling
            try:
                chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
            except Exception as e:
                logger.warning(f"Error loading chat history: {e}")
                chat_history = []

            # Conversational chain using RunnablePassthrough
            chain = (
                RunnablePassthrough.assign(
                    context=lambda x: "\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["input"])])
                )
                | prompt
                | self.llm
                | StrOutputParser()
            )
            # Invoke the chain with the question and chat history
            response = chain.invoke({"input": question, "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])})

            # Save to memory *after* the LLM call
            self.memory.save_context({"input": question}, {"answer": response})
            return response

        except Exception as e:
            logger.exception(f"Error querying RAG system: {e}")  # More detailed logging
            return ""