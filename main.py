from agent.agent import QAAgent
from base.constants import Constants
from config import HF_TOKEN, GOOGLE_API_KEY
from google.genai import types
import time
from rag.rag_content_retriever import RAGContentRetriever, RAGCorpusManager
from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool
from utils.model_loader import ModelLoader
from utils.dataset_utils import DataSetUtil
from web_search.web_search import WebSearch
from agent.tools.web_search_tool import WebSearchTool

"""
* I need to give the agent a tool to call so that it can clear the knowledge base used by RAG_TOOL
* FIx the way the tool is being called (args are not passed correctly)
* I need to add capability to process a given document and use rag to answer questions based on that document.
  - I actually can modify the rag tool to accept a document as input, process it, add it to the index, and then answer questions based on it.
"""

if __name__ == "__main__":


    model_loader = ModelLoader()
    tokenizer = model_loader.load_hf_tokenizer(hugging_face_token = HF_TOKEN)
    cross_encoder = model_loader.load_hf_cross_encoder(hugging_face_token = HF_TOKEN)
    sentence_transformer = model_loader.load_sentence_embedding_model(hugging_face_token = HF_TOKEN)

    generative_model = model_loader.load_gemini_generative_model(google_api_key = GOOGLE_API_KEY, config = types.GenerateContentConfig(), model_name = Constants.ModelNames.Gemini.GEMINI_2_5_PRO)
    # generative_model = model_loader.load_hf_generative_model(hugging_face_token = HF_TOKEN)


    rag_corpus_manager_for_knowledge_base = RAGCorpusManager(sentence_transformer = sentence_transformer)
    rag_corpus_manager_for_web_search_tool = RAGCorpusManager(sentence_transformer = sentence_transformer)
    dataset_util = DataSetUtil()
    knowledge_base = dataset_util.load_qa_dataset()
    rag_corpus_manager_for_knowledge_base.add_update_data_and_index(knowledge_base)
    rag_content_retriever = RAGContentRetriever(cross_encoder, generative_model)
    rag_tool = RagTool(rag_corpus_manager_for_knowledge_base, rag_content_retriever)
    web_search_tool = WebSearchTool(rag_content_retriever=rag_content_retriever, rag_corpus=rag_corpus_manager_for_web_search_tool)
    qa_agent = QAAgent(rag_tool, web_search_tool, generative_model)

    """
    print("\n\n\n")
    query = "What does Varian Medical Systems do? Tell me briefly."
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")

    query = "Based on the knowledge base documents, who is the president of Notre Dame university?"
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")
    """

    query = """
            Summarize the following text: Natural language processing (NLP) is a subfield of 
            artificial intelligence (AI) focused on the interaction between computers and 
            humans through natural language. The ultimate objective of NLP is to enable 
            computers to understand, interpret, and generate human language in a way that 
            is valuable. Applications of NLP include language translation, sentiment analysis, 
            speech recognition, and chatbots.
    """
    
    """
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")

    query = "What is todays date?"
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")

    query = "According to our uploaded documents, When was Notre Dame University President last changed?"
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")


    query = "What is latest Gemini model?"
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")
    """

    # The agent currently fails to answer this! I can make it iteratively to fix the issue (feed back the response to the agent together with the tool and ask it to try again)
    # It may still not work as the web-search-tool fails to grab the correct data anyways
    query = "What was the stock price of Google on November 25, 2025 according to the internet?"
    print(f"{query}\n {qa_agent.chat(query)}")
    print(f"\n\n{'_'*50}\n\n")


    # I need to come up with an example that model fails to respond; then it figures out that if do it iteratively, it can find the answer using the web search tool