from rag.rag_content_retriever import RAGContentRetriever, RAGCorpusManager
from typing import List, Union, Dict, Tuple


class RagTool:
    """High-level convenience interface for RAG corpus management and querying.

    Attributes:
        corpus_manager (RAGCorpusManager): Manages storage, chunking, embeddings, and indexing
            of documents added to the RAG system.
        content_retriever (RAGContentRetriever): Performs similarity search, reranking and
            response generation over the managed corpus.
    """

    def __init__(self, corpus_manager: RAGCorpusManager, content_retriever: RAGContentRetriever):
        """Initialize the RAG tool wrapper.

        Args:
            corpus_manager (RAGCorpusManager): Instance responsible for persisting and indexing documents.
            content_retriever (RAGContentRetriever): Retrieval + generation component operating over the corpus.
        """
        self.corpus_manager = corpus_manager
        self.content_retriever = content_retriever

    def add_docs(self, docs: List[Union[str, Dict]]) -> None:
        """Add one or more documents to the RAG corpus.

        Each element may be either:
          * str: Raw textual content; will be wrapped into a minimal dict {'context': <text>}.
          * dict: A pre-structured document object expected to contain at least a 'context' key
            plus any optional metadata the corpus manager can leverage.

        Documents are immediately indexed (embeddings + vector store) via the corpus manager.

        Args:
            docs (List[Union[str, Dict]]): Raw strings OR structured dicts representing documents.

        Notes:
            This method currently performs a simple type check and passes dict objects through
            verbatim. Future enhancements could include schema validation, batching, and duplicate
            detection.
        """
        for doc in docs:
            if isinstance(doc, str):
                self.corpus_manager.add_update_data_and_index({"context": doc})
            else:
                self.corpus_manager.add_update_data_and_index(doc)

    def use_tool(self, query: str, max_results: int = 5) -> List[str]:
        """Execute a retrieval + generation cycle over the current corpus.

        Args:
            query (str): User's query
            max_results (int): Maximum number of top similar items to  be considered for generation.

        Returns:
            List[str]: context: Aggregated, merged context passages surfaced as most relevant.

        See Also:
            RAGContentRetriever.find_query_response: for configurable ranking parameters.
        """
        top_results = self.content_retriever.find_top_similar_items(
            self.corpus_manager, query, cross_encoder_top_k=max_results
        )
        if not top_results:
            return [""]
        return top_results[:][0]
