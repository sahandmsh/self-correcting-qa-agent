from agent.agent import QAAgent
from base.constants import Constants
from config import HF_TOKEN, GOOGLE_API_KEY
from google.genai import types
from rag.rag_content_retriever import RAGContentRetriever, RAGCorpusManager
from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool
from utils.model_loader import ModelLoader
from utils.dataset_utils import DataSetUtil
from agent.tools.web_search_tool import WebSearchTool

if __name__ == "__main__":

    # ============================================================================
    # STEP 1: Load Models and Components
    # ============================================================================
    # Initialize the model loader utility to load various AI models
    model_loader = ModelLoader()

    # Load HuggingFace tokenizer for text processing
    tokenizer = model_loader.load_hf_tokenizer(hugging_face_token=HF_TOKEN)

    # Load cross-encoder model for re-ranking retrieved documents
    cross_encoder = model_loader.load_hf_cross_encoder(hugging_face_token=HF_TOKEN)

    # Load sentence transformer model for generating text embeddings
    sentence_transformer = model_loader.load_sentence_embedding_model(hugging_face_token=HF_TOKEN)

    # Configure and load the Gemini generative model for LLM-based responses
    # Set temperature=0.7 for balanced creativity and add system instructions
    gemini_config = types.GenerateContentConfig(
        temperature=0.7, system_instruction=Constants.Instructions.System.SYSTEM_INSTRUCTION
    )
    generative_model = model_loader.load_gemini_generative_model(
        google_api_key=GOOGLE_API_KEY,
        config=gemini_config,
        model_name=Constants.ModelNames.Gemini.GEMINI_2_5_PRO,
    )

    # ============================================================================
    # STEP 2: Initialize RAG Components
    # ============================================================================
    # Create a RAG corpus manager for the knowledge base (local documents)
    rag_corpus_manager_for_knowledge_base = RAGCorpusManager(
        sentence_transformer=sentence_transformer
    )

    # Create a separate RAG corpus manager for web search results
    rag_corpus_manager_for_web_search_tool = RAGCorpusManager(
        sentence_transformer=sentence_transformer
    )

    # Load the knowledge base dataset (squd databas)
    dataset_util = DataSetUtil()
    knowledge_base = dataset_util.load_qa_dataset()

    # Index the knowledge base documents into the corpus manager
    rag_corpus_manager_for_knowledge_base.add_update_data_and_index(knowledge_base)

    # Initialize the content retriever with cross-encoder for re-ranking
    rag_content_retriever = RAGContentRetriever(cross_encoder, generative_model)

    # ============================================================================
    # STEP 3: Set Up Agent Tools
    # ============================================================================
    # Create RAG tool for querying the knowledge base
    rag_tool = RagTool(rag_corpus_manager_for_knowledge_base, rag_content_retriever)

    # Create web search tool for internet queries
    web_search_tool = WebSearchTool(
        rag_content_retriever=rag_content_retriever,
        rag_corpus=rag_corpus_manager_for_web_search_tool,
    )

    # Initialize the QA agent with both tools and the generative model
    qa_agent = QAAgent(rag_tool, web_search_tool, generative_model)

    # ============================================================================
    # STEP 4: Test Queries - General Knowledge / Web Search
    # ============================================================================
    # Test the agent's ability to use web search/general knowledge for information
    query = "What is Agentic AI? Tell me in a sentence."
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # Test agent's ability to use web search for information
    query = "What is latest Gemini model?"
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # Test agent's ability to use web search for information
    query = "What was the stock price of Google on November 25, 2025 according to the internet?"
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # ============================================================================
    # STEP 5: Test Queries - Knowledge Base Specific
    # ============================================================================
    # Test querying the local knowledge base
    query = "Based on the knowledge base documents, who is the president of Notre Dame university?"
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # Test another knowledge base specific query
    query = "According to our uploaded documents, When was Notre Dame University President last changed?"
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # ============================================================================
    # STEP 6: Test Queries - Tool Selection Behavior
    # ============================================================================
    # These queries test the agent's ability to select the appropriate tool
    # based on how the question is phrased

    # Explicitly request knowledge base lookup
    query = "According to the knowledge base, who was the first president of Notre Dame University?"
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # Explicitly request model's internal knowledge (no tool use)
    query = "Who was the first president of Notre Dame University? Answer using your own knowledge."
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")

    # Explicitly request web search
    query = "Who was the first president of Notre Dame University? Search the web."
    print(f"{query}\n {qa_agent.chat(query)}\n{'_'*50}\n")
