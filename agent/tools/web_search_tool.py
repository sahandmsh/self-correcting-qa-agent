from rag.rag_corpus_manager import RAGCorpusManager
from rag.rag_content_retriever import RAGContentRetriever
from typing import List
from utils.web_search import WebSearch


class WebSearchTool:
    """Perform a live web search then run RAG retrieval over the fetched results.

    Attributes:
        web_search (WebSearch): Component used to execute external web queries.
        rag_corpus (RAGCorpusManager): Manages temporary storage/indexing of web results.
        rag_content_retriever (RAGContentRetriever): Performs similarity search and ranking.
    """

    def __init__(self, rag_content_retriever: RAGContentRetriever, rag_corpus: RAGCorpusManager):
        """Initialize the web search tool.

        Args:
            rag_content_retriever (RAGContentRetriever): Retrieval component for ranking passages.
            rag_corpus (RAGCorpusManager): Corpus manager used as ephemeral store for web results.
        """
        self.web_search = WebSearch()
        self.rag_corpus = rag_corpus
        self.rag_content_retriever = rag_content_retriever

    def use_tool(self, query: str, max_web_pages: int = 5, max_top_passages: int = 3) -> List[str]:
        """Execute web search + RAG retrieval for a query.

        Process:
            1. Fetch up to `max_web_pages` web results and index them (chunking/embedding handled downstream).
            2. Run similarity search to rank indexed passages.
            3. Return the top `max_top_passages` scoring passages and their relevance scores.

        Args:
            query (str): User query
            max_web_pages (int): Maximum number of web search results to fetch.
            max_top_passages (int): Number of top passages to retrieve from RAG.

        Returns:
            List[str]: Top passages retrieved from web search results.
        """
        self.rag_corpus.add_update_data_and_index(
            self.web_search.search(query, max_results=max_web_pages), text_content_key="content"
        )
        top_results = self.rag_content_retriever.find_top_similar_items(
            self.rag_corpus, query, cross_encoder_top_k=max_top_passages
        )
        if not top_results:
            return [""]
        return top_results[:][0]
