from base.constants import Constants
from rag.rag_corpus_manager import RAGCorpusManager
from typing import List

import faiss
import heapq
import torch


class RAGContentRetriever:
  """
  Handles end-to-end answer retrieval in a Retrieval-Augmented Generation (RAG) system.

  This class integrates three main components:
    1. A **bi-encoder** (via FAISS index) for fast passage retrieval.
    2. A **cross-encoder** for re-ranking retrieved passages by semantic relevance.
    3. A **generative model** (e.g., Gemini) for producing natural language answers
       conditioned on the retrieved context.
  """
  def __init__(self, cross_encoder, generative_model):
    """
        Initializes the RAGContentRetriever class with cross-encoder and generative model.

        Args:
            cross_encoder (CrossEncoder): Pretrained cross-encoder model from sentence-transformers.
            generative_model: Gemini generative model with prompt (str) as input and generated text (str) as output.
        """
    self._cross_encoder = cross_encoder
    self._generative_model = generative_model


  def _merge_passage_chunks_and_scores(self, chunks_list):
    """
    Merges overlapping text chunks and computes their average relevance score.

    Args:
        chunks_list (list[tuple]): List of tuples, each containing:
            - start_word_index (int): Start position of the chunk in the original text.
            - chunk_text (str): Chunk text content.
            - score (float): Relevance score assigned by the cross-encoder.

    Returns:
        tuple:
            str: Merged text string from the chunks.
            float: Average relevance score across the merged chunks.
    """
    merged_text_list = []
    chunk_score = 0
    for i in range(len(chunks_list)):
        if i < len(chunks_list) - 1 and chunks_list[i+1][0] <= chunks_list[i][0] + len(chunks_list[i][1]) - 1:
            merged_text_list.append(chunks_list[i][1][ : chunks_list[i+1][0] - chunks_list[i][0]])
        else:
            merged_text_list.append(chunks_list[i][1])
        chunk_score += chunks_list[i][2]
    return " ".join(merged_text_list), chunk_score/len(chunks_list)


  def _biencoder_find_top_similar_items(self, query: str, top_k: int, rag_corpus: RAGCorpusManager):
    """
    Retrieves top-k semantically similar passages using the FAISS bi-encoder index.

    Args:
        query (str): Query text to search for.
        top_k (int): Number of top results to retrieve.
        rag_corpus (RAGCorpusManager): Corpus manager containing FAISS index and embeddings.

    Returns:
        list[int]: Indices of the top-k most similar passage chunks in the corpus.
    """
    query_embedding = rag_corpus.sentence_transformer.encode([query], convert_to_numpy = True)
    faiss.normalize_L2(query_embedding)
    _, indices = rag_corpus.faiss_index.search(query_embedding, top_k)
    return indices[0].tolist()


  def _cross_encoder_find_top_similar_items(self, query: str, rag_corpus: RAGCorpusManager, passage_chunk_indices: List[int], top_k: int = 5, batch_size: int = 64, max_length: int = 512):
    """
    Re-ranks retrieved passages using a cross-encoder for more precise semantic relevance.

    Args:
        query (str): Input query text.
        rag_corpus (RAGCorpusManager): Corpus manager containing chunked data and metadata.
        passage_chunk_indices (list[int]): Indices of passage chunks retrieved by the bi-encoder.
        top_k (int, optional): Number of top passages to keep after re-ranking. Defaults to 5.
        batch_size (int, optional): Number of passage pairs to process in a single batch. Defaults to 64.
        max_length (int, optional): Maximum token length per input pair. Defaults to 512.

    Returns:
        list[tuple]: List of tuples containing merged passage text and corresponding average score,
        sorted by descending relevance.
    """

    self._cross_encoder.eval()
    scores = []

    with torch.no_grad():
        for start in range(0, len(passage_chunk_indices), batch_size):
            batch_indices = passage_chunk_indices[start:start + batch_size]
            batch_passages = [rag_corpus.chunked_data[i] for i in batch_indices]
            batch_pairs = [[query, passage] for passage in batch_passages]
            batch_scores = self._cross_encoder.predict(batch_pairs)
            scores.extend(zip(batch_indices, batch_scores))
    most_related_items_index_score_pairs = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    top_chunks_idx_to_metadata = {}
    top_k_results = []
    for index, score in most_related_items_index_score_pairs:
      document_id = rag_corpus.chunked_data_metadata[index]['document_id']
      start_word_index = rag_corpus.chunked_data_metadata[index]['start_word_index']
      top_chunks_idx_to_metadata.setdefault(document_id, [])
      heapq.heappush(top_chunks_idx_to_metadata[document_id], (start_word_index, rag_corpus.chunked_data[index], score))
    for chunks_list in top_chunks_idx_to_metadata.values():
            top_k_results.append(self._merge_passage_chunks_and_scores(chunks_list))
    return top_k_results


  def find_top_similar_items(self, rag_corpus: RAGCorpusManager, query: str, biencoder_top_k: int = 100, cross_encoder_top_k: int = 5, cross_encoder_batch_size: int = 16, max_cross_encoder_token_length: int = 512):
    """
    Retreives top similar passages to the query (first finding top candidates using biencoder, then rearranging them using cross-encoder).

    Args:
      rag_corpus (RAGCorpusManager): Corpus manager containing documents, embeddings, and metadata.
      query (str): User query string.
      biencoder_top_k (int, optional): Number of passages retrieved by the bi-encoder. Defaults to 100.
      cross_encoder_top_k (int, optional): Number of passages kept after cross-encoder re-ranking. Defaults to 5.
      cross_encoder_batch_size (int, optional): Batch size for cross-encoder inference. Defaults to 16.
      max_cross_encoder_token_length (int, optional): Maximum token length for cross-encoder inputs. Defaults to 512.

    Returns:
        list[tuple]: List of tuples containing merged passage text and corresponding average score,
        sorted by descending relevance.
    """
    biencoder_top_passage_chunk_indices = self._biencoder_find_top_similar_items(query, biencoder_top_k, rag_corpus)
    cross_encoder_top_similar_items = self._cross_encoder_find_top_similar_items(query, rag_corpus, biencoder_top_passage_chunk_indices, cross_encoder_top_k, cross_encoder_batch_size, max_cross_encoder_token_length)
    return cross_encoder_top_similar_items


  def find_query_response(
        self, 
        rag_corpus: RAGCorpusManager, 
        query: str, 
        biencoder_top_k: int = 100, 
        cross_encoder_top_k = 5, 
        cross_encoder_batch_size: int = 16, 
        max_cross_encoder_token_length: int = 512,
        rag_based_instruction: str = Constants.Instructions.RAG_CONTEXT_BASED_INSTRUCTION
      ):
    """
    Retrieves the most relevant context and generates a natural language response for a query.

    This method performs the following steps:
      1. Uses the bi-encoder FAISS index to find top-k relevant passages.
      2. Re-ranks those passages using a cross-encoder for better precision.
      3. Merges overlapping chunks.
      4. Constructs a prompt combining the query and top contexts.
      5. Generates a response using the generative model.

    Args:
        rag_corpus (RAGCorpusManager): Corpus manager containing documents, embeddings, and metadata.
        query (str): User query string.
        biencoder_top_k (int, optional): Number of passages retrieved by the bi-encoder. Defaults to 100.
        cross_encoder_top_k (int, optional): Number of passages kept after cross-encoder re-ranking. Defaults to 5.
        cross_encoder_batch_size (int, optional): Batch size for cross-encoder inference. Defaults to 16.
        max_cross_encoder_token_length (int, optional): Maximum token length for cross-encoder inputs. Defaults to 512.

    Returns:
        tuple:
            str: Combined context text used for response generation.
            str: Generated answer from the generative model.
    """
    top_similar_items = self.find_top_similar_items(rag_corpus, query, biencoder_top_k, cross_encoder_top_k, cross_encoder_batch_size, max_cross_encoder_token_length)
    context = "\nContext: ".join([passage_and_score[0] for passage_and_score in top_similar_items])
    prompt = rag_based_instruction.format(context = context, query = query)
    response = self._generative_model(prompt)
    return context, response
