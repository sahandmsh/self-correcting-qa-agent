from base.constants import Constants
from rag.rag_corpus_manager import RAGCorpusManager
from rag.rag_content_retriever import RAGContentRetriever
from web_search.web_search import WebSearch
from sentence_transformers import SentenceTransformer

class WebSearchTool:
    """ A tool that performs web search and retrieves answers using RAG.
    Attributes:
        web_search (WebSearch): The web search component
        rag_answer_retriever (RAGContentRetriever): The RAG answer retrieval component
    """
    def __init__(self, rag_content_retriever: RAGContentRetriever, rag_corpus: RAGCorpusManager):
        self.web_search = WebSearch()
        self.rag_corpus = rag_corpus
        self.rag_content_retriever = rag_content_retriever


    def use_tool(self, query: str) -> str:
        self.rag_corpus._clear_corpus()
        self.rag_corpus.add_update_data_and_index(self.web_search.search(query), text_content_key = 'content')
        top_results = self.rag_content_retriever.find_top_similar_items(self.rag_corpus, query)
        return top_results[:][0]