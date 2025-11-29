from rag.rag_content_retriever import RAGContentRetriever, RAGCorpusManager
from typing import List

class RagTool:
    """ RAG Tool to interact with RAG components
    Attributes:
        corpus_manager (RAGCorpusManager): The RAG corpus manager
        content_retriever (RAGContentRetriever): The RAG answer retrieval component
    """
    def __init__(self, corpus_manager: RAGCorpusManager, content_retriever: RAGContentRetriever):
        """ Initializes the RagTool
        Args:
            corpus_manager (RAGCorpusManager): The RAG corpus manager
            content_retriever (RAGContentRetriever): The RAG answer retrieval component
        Returns:
            None
        """
        self.corpus_manager = corpus_manager
        self.content_retriever = content_retriever

    def add_docs(self, docs: List):
        """ Adds documents to the corpus manager
        Args:
            docs (List): List of documents to add
        Returns:
            None
        """
        # I need to modify this for more friendly add docs method.
        for doc in docs:
            if isinstance(doc, str):
                self.corpus_manager.add_update_data_and_index({'context': doc})
            else:
                self.corpus_manager.add_update_data_and_index(doc)
        

    def use_tool(self, query: str):
        """ Uses the RAG tool to find the answer to a query
        Args:
            query (str): The query to find the answer for
        Returns:
            str: The answer to the query
        """
        return self.content_retriever.find_query_response(self.corpus_manager, query)