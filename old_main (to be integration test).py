from config import HF_TOKEN, GOOGLE_API_KEY
from google.genai import types
import time
from rag.rag_content_retriever import RAGContentRetriever, RAGCorpusManager
from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool
from utils.model_loader import ModelLoader
from utils.dataset_utils import DataSetUtil


if __name__ == "__main__":
    """ Example usage of RAG implementation with Gemini model and Hugging Face models for tokenizer, cross-encoder, and sentence transformer. """
    """ Example usage of Web Search Tool """
    web_search_tool = WebSearchTool()
    print(web_search_tool.use_tool("What is agentic ai?"))
    """Following is an example use case of the above RAG implementation"""
    """1. The knowledge base is created by sampling the Squad dataset."""
    dataset_util = DataSetUtil()
    knowledge_base = dataset_util.load_qa_dataset()
    """2. Load the cross encoder and sentence transformer (for biencoder)."""
    model_loader = ModelLoader()
    cross_encoder = model_loader.load_hf_cross_encoder(hugging_face_token = HF_TOKEN)
    sentence_transformer = model_loader.load_sentence_embedding_model(hugging_face_token = HF_TOKEN)
    """### 3. Create the rag_corpus object and index the given knowledge base after chunking its text and calculate sentence embeddings."""
    rag_corpus = RAGCorpusManager(sentence_transformer = sentence_transformer)
    rag_corpus.add_update_data_and_index(knowledge_base)
    """### 4. The next step is to load a generative language model."""
    config = types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens= 1024,
        system_instruction="Return plain text only, no markdown, no special characters. Only used provided 'contexts' to generate response. Always Return 'I don't know' if the response is not found in the context."
    )
    generative_model = model_loader.load_gemini_generative_model(google_api_key = GOOGLE_API_KEY, config = config)
    """### 5. Then, the rag_content_retriever object is created."""
    rag_content_retriever = RAGContentRetriever(cross_encoder, generative_model)
    """### 6. The retriever is now ready to be used. Following are a few example use cases of finding the answer of the given question using the developed RAG pipeline."""
    question = "Where is Notre Dame?"
    response = rag_content_retriever.find_query_response(rag_corpus, question)[1]
    print(f"{question} {response}")

    time.sleep(15)
    question = "Where is Indiana?"
    rag_content_retriever.find_query_response(rag_corpus, question)[1]
    print(f"{question} {response}")

    time.sleep(15)
    question = "Who is Notre Dome President?"
    response = rag_content_retriever.find_query_response(rag_corpus, question)[1]
    print(f"{question} {response}")

    time.sleep(15)
    question = "Who is Notre Dome President in 2008?"
    response = rag_content_retriever.find_query_response(rag_corpus, question)[1]
    print(f"{question} {response}")

    """ NOTE: The RAG pipeline may fail to find an answer to a question in the following cases:

    * The answer does not exist in the underlying knowledge base or database.

    * The RAG retriever fails to retrieve the relevant context for the query.

    * The generative model fails to produce the correct answer even when the relevant context is retrieved.
    """
    time.sleep(15)
    question = "Who is 16th president of Notre Dome?"
    response = rag_content_retriever.find_query_response(rag_corpus, question)[1]
    print(f"{question} {response}")
