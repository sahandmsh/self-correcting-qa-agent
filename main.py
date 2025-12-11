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
from utils.web_search import WebSearch

if __name__ == "__main__":
    # ============================================================================
    # STEP 1: Load Models and Components
    # ============================================================================
    # Initialize the model loader utility to load various AI models
    model_loader = ModelLoader()

    # Load cross-encoder model for re-ranking retrieved documents
    cross_encoder = model_loader.load_hf_cross_encoder(hugging_face_token=HF_TOKEN)

    # Load sentence transformer model for generating text embeddings
    sentence_transformer = model_loader.load_sentence_embedding_model(hugging_face_token=HF_TOKEN)

    # Configure and load the Gemini generative model for LLM-based responses
    gemini_config = types.GenerateContentConfig(
        temperature=0.7, system_instruction=Constants.Instructions.AgenticAI.SYSTEM_INSTRUCTIONS
    )
    generative_model = model_loader.load_gemini_generative_model(
        google_api_key=GOOGLE_API_KEY,
        config=gemini_config,
        model_name=Constants.ModelNames.Gemini.GEMINI_2_5_FLASH,
    )

    critique_config = types.GenerateContentConfig(
        temperature=0.7,
        system_instruction=Constants.Instructions.Critique.SYSTEM_INSTRUCTIONS,
    )
    critiquer_model = model_loader.load_gemini_generative_model(
        google_api_key=GOOGLE_API_KEY,
        config=critique_config,
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
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Test agent's ability to use web search for information
    query = "What is latest Gemini model?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Test agent's ability to use web search for information
    query = "What was the stock price of Google on November 25, 2025 according to the internet?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # ============================================================================
    # STEP 5: Test Queries - Knowledge Base Specific
    # ============================================================================
    # Test querying the local knowledge base
    query = "Based on the knowledge base documents, who is the president of Notre Dame university?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Test another knowledge base specific query
    query = "According to our uploaded documents, When was Notre Dame University President last changed?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # ============================================================================
    # STEP 6: Test Queries - Tool Selection Behavior
    # ============================================================================
    # These queries test the agent's ability to select the appropriate tool
    # based on how the question is phrased

    # Explicitly request knowledge base lookup
    query = "According to the knowledge base, who was the first president of Notre Dame University?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Explicitly request model's internal knowledge (no tool use)
    query = "Who was the first president of Notre Dame University? Answer using your own knowledge."
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Explicitly request web search
    query = "Who was the first president of Notre Dame University? Search the web."
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # ============================================================================
    # STEP 7: Additional Test Queries - AI & Temporal Awareness
    # ============================================================================
    # Latest model release (should use web search)
    query = "What is the latest version of GPT released by OpenAI?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Recent AI news (should use web search - current events)
    query = "What are the major AI announcements from Google in 2025?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # Specific date query (should use web search)
    query = "When was Claude 3.5 Sonnet released?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")

    # General AI knowledge (no tool needed - within knowledge cutoff)
    query = "What is the transformer architecture?"
    response, context = qa_agent.agent_chat(query)
    print(f"query: {query}\nresponse: {response}\n")
    critique_feedback = critiquer_model(
        f"query: {query}\nAgentic AI response: {response}\n{f'Agentic AI retrieved context using tools: {context}' if context else ''}"
    )
    print(f"Critique Feedback: {critique_feedback}\n\n {'_'*50}\n")
