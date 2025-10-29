from datasets import load_dataset
from google import genai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

import faiss
import heapq
import numpy as np
import torch


class RAGCorpusManager:
  """
  Manages a Retrieval-Augmented Generation (RAG) corpus.

  This class handles text normalization, deduplication, chunking, embedding generation,
  and FAISS index management for efficient document retrieval in RAG systems.
  """


  def __init__(self, sentence_transformer: SentenceTransformer, max_data_chunk_len: int = 300, data_chunk_stride: int = 75):
    """
    Initializes the RAGCorpusManager.

    Args:
        hugging_face_token (str): Authentication token for accessing Hugging Face models.
        serntence_transformer (SentenceTransformer): The sentence transformer model from Hugging Face.
            used for embedding generation. Defaults to "all-MiniLM-L6-v2".
        max_data_chunk_len (int, optional): Maximum number of words per chunk. Defaults to 300.
        data_chunk_stride (int, optional): Overlap between consecutive chunks in words. Defaults to 75.
    """
    self.raw_dataset = []
    self._unique_text_set = set()
    self.chunked_data = []
    self.chunked_data_metadata = []
    self.sentence_embeddings = []
    self.faiss_indices = []
    self.sentence_transformer = sentence_transformer
    self.max_data_chunk_len = max_data_chunk_len
    self.data_chunk_stride = data_chunk_stride
    if self.max_data_chunk_len <= 0:
        raise ValueError("Invalid chunk length; it should be a positive value")
    if self.data_chunk_stride < 0 or self.data_chunk_stride >= self.max_data_chunk_len:
        raise ValueError("Invalid stride value; make sure that 0<= data_chunk_stride < max_data_chunk_len")


  def _normalize_text(self, text: str):
    """
    Normalizes input text by trimming extra spaces and line breaks.

    Args:
        text (str): Input text string.

    Returns:
        str: Normalized text with single spaces between words.
    """
    return " ".join(text.strip().split())


  def _find_new_data_entries(self, new_dataset: List[dict]):
    """
    Filters out duplicate entries from the provided dataset.

    Args:
        new_dataset (list[dict]): List of dataset entries.

    Returns:
        list[dict]: List of unique new entries not already available in the corpus.
    """
    new_unique_dataset = []
    for item in new_dataset:
      normalized_text = self._normalize_text(item.get('context', ""))
      if normalized_text and normalized_text not in self._unique_text_set:
        self._unique_text_set.add(normalized_text)
        new_unique_dataset.append(item)
    return new_unique_dataset


  def _chunk_data(self, dataset: List[dict]):
    """
    Splits dataset entries into overlapping text chunks based on 'word' count.

    Args:
        dataset (list[dict]): List of data items

    Returns:
        tuple:
            list[str]: List of text chunks.
            list[dict]: List of metadata dictionaries for each chunk, including:
                - 'source_id': ID of the source document.
                - 'data_chunk_id': Sequential chunk ID.
                - 'start_word_index': Starting word index in the source text.
    """
    data_chunks = []
    data_chunks_metadata = []
    for item in dataset:
        text = item.get('context', "")
        if not text:
            continue
        words_list = text.split()
        if len(words_list)<=self.max_data_chunk_len:
            data_chunks.append(text)
            data_chunks_metadata.append({'source_id': item.get('id', "NA"), 'data_chunk_id': 0, 'start_word_index': 0})
            continue
        start_word_index = 0
        data_chunk_id = 0
        while start_word_index < len(words_list):
            chunk = words_list[start_word_index:start_word_index + self.max_data_chunk_len]
            data_chunks.append(" ".join(chunk))
            data_chunks_metadata.append({
                'source_id': item.get('id', "NA"),
                'data_chunk_id': data_chunk_id,
                'start_word_index': start_word_index
            })
            if start_word_index + self.max_data_chunk_len >= len(words_list):
                break
            start_word_index += self.max_data_chunk_len - self.data_chunk_stride
            data_chunk_id += 1
    return data_chunks, data_chunks_metadata


  def _calculate_sentence_embeddings(self, data_chunks: List[str], batch_size: int = 64):
    """
    Computes sentence embeddings for a list of text chunks.

    Args:
        data_chunks (list[str]): List of text chunks to embed.
        batch_size (int, optional): Number of chunks processed per batch. Defaults to 64.

    Returns:
        np.ndarray: Array of embeddings corresponding to each input chunk.
    """
    sentence_embeddings = self.sentence_transformer.encode(
        data_chunks,
        batch_size = batch_size,
        show_progress_bar = True,
        convert_to_numpy = True
    )
    return sentence_embeddings


  def _create_update_faiss_index(self, embeddings: np.ndarray):
    """
    Creates or updates the FAISS index with new embeddings.

    Args:
        embeddings (np.ndarray): New embeddings to add to the FAISS index.

    Notes:
        - Uses cosine similarity (inner product) as the metric.
        - Normalizes embeddings before indexing.
        - Creates the FAISS index object if it does not already exist.
    """
    faiss.normalize_L2(embeddings)
    if not hasattr(self, "faiss_index"):
      embedding_dim = embeddings.shape[1]
      faiss.normalize_L2(embeddings)
      self.faiss_index = faiss.IndexFlatIP(embedding_dim)
    self.faiss_index.add(embeddings)


  def add_update_data_and_index(self, dataset: List[dict]):
    """
    Adds new data entries to the corpus and updates the FAISS index.

    This method performs the following steps:
      1. Removes duplicate entries.
      2. Splits texts into overlapping chunks.
      3. Computes embeddings for the new chunks.
      4. Updates the FAISS index.
      5. Extends the internal corpus data structures.

    Args:
        dataset (list[dict]): New dataset entries to add.

    Returns:
        None
    """
    new_data_entries = self._find_new_data_entries(dataset)
    if not new_data_entries:
      return
    new_data_chunks, new_data_chunks_metadata = self._chunk_data(new_data_entries)
    new_embeddings = self._calculate_sentence_embeddings(new_data_chunks)
    self._create_update_faiss_index(new_embeddings)
    self.raw_dataset.extend(new_data_entries)
    self.chunked_data.extend(new_data_chunks)
    self.chunked_data_metadata.extend(new_data_chunks_metadata)


class RAGAnswerRetrieval:
  """
  Handles end-to-end answer retrieval in a Retrieval-Augmented Generation (RAG) system.

  This class integrates three main components:
    1. A **bi-encoder** (via FAISS index) for fast passage retrieval.
    2. A **cross-encoder** for re-ranking retrieved passages by semantic relevance.
    3. A **generative model** (e.g., Gemini) for producing natural language answers
       conditioned on the retrieved context.
  """
  def __init__(self, tokenizer, cross_encoder, generative_model):
    """
        Initializes the RAGAnswerRetrieval class with tokenizer, cross-encoder, and generative model.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for the cross-encoder.
            cross_encoder (AutoModelForSequenceClassification): Pretrained cross-encoder model.
            generative_model: Gemini generative model with prompt (str) as input and generated text (str) as output.
        """
    self._tokenizer = tokenizer
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


  def _biencoder_find_top_similar_items(self, query, top_k, rag_corpus: RAGCorpusManager):
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
            batch_pairs = list(zip([query] * len(batch_passages), batch_passages))
            inputs = self._tokenizer(batch_pairs, padding=True, truncation='only_second', max_length=max_length, return_tensors="pt")
            outputs = self._cross_encoder(**inputs)
            logits = outputs.logits
            batch_scores = logits.view(-1).tolist()
            scores.extend(zip(batch_indices, batch_scores))
    most_related_items_index_score_pairs = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    top_chunks_idx_to_metadata = {}
    top_k_results = []
    for index, score in most_related_items_index_score_pairs:
      passage_id = rag_corpus.chunked_data_metadata[index]['source_id']
      start_word_index = rag_corpus.chunked_data_metadata[index]['start_word_index']
      top_chunks_idx_to_metadata.setdefault(passage_id, [])
      heapq.heappush(top_chunks_idx_to_metadata[passage_id], (start_word_index, rag_corpus.chunked_data[index], score))
    for key, chunks_list in top_chunks_idx_to_metadata.items():
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


  def find_query_response(self, rag_corpus: RAGCorpusManager, query: str, biencoder_top_k: int = 100, cross_encoder_top_k = 5, cross_encoder_batch_size: int = 16, max_cross_encoder_token_length: int = 512):
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
    prompt = f"Question: {query}\nContext: {context}"
    response = self._generative_model(prompt)
    return context, response
