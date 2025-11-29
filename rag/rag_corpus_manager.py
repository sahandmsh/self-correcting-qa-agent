from base.constants import Constants
from sentence_transformers import SentenceTransformer
from typing import List

import faiss
import numpy as np

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
    self._document_id_tracker = 0
    self.raw_dataset = []
    self._unique_text_set = set()
    self.chunked_data = []
    self.chunked_data_metadata = []
    self.sentence_embeddings = []
    self.sentence_transformer = sentence_transformer
    self.max_data_chunk_len = max_data_chunk_len
    self.data_chunk_stride = data_chunk_stride
    if self.max_data_chunk_len <= 0:
        raise ValueError("Invalid chunk length; it should be a positive value")
    if self.data_chunk_stride < 0 or self.data_chunk_stride >= self.max_data_chunk_len:
        raise ValueError("Invalid stride value; make sure that 0<= data_chunk_stride < max_data_chunk_len")


  def _clear_corpus(self):
    """
    Clears the existing corpus data and resets internal states.

    This method removes all stored documents, chunked data, embeddings,
    and resets the document ID tracker and unique text set.
    """
    self._document_id_tracker = 0
    self.raw_dataset = []
    self._unique_text_set = set()
    self.chunked_data = []
    self.chunked_data_metadata = []
    self.sentence_embeddings = []
    if hasattr(self, "faiss_index"):
        del self.faiss_index


  def _normalize_text(self, text: str):
    """
    Normalizes input text by trimming extra spaces and line breaks.

    Args:
        text (str): Input text string.

    Returns:
        str: Normalized text with single spaces between words.
    """
    return " ".join(text.strip().split())


  def _find_new_data_entries(self, new_dataset: List[dict], text_content_key: str = 'context'):
    """
    Filters out duplicate entries from the provided dataset.

    Args:
        new_dataset (list[dict]): List of dataset entries.
        text_content_key (str, optional): Key to extract text content from each data item. Defaults to 'context'.

    Returns:
        list[dict]: List of unique new entries not already available in the corpus.
    """
    new_unique_dataset = []
    for item in new_dataset:
      normalized_text = self._normalize_text(item.get(text_content_key, ""))
      if normalized_text and normalized_text not in self._unique_text_set:
        self._unique_text_set.add(normalized_text)
        new_unique_dataset.append(item)
    return new_unique_dataset


  def _chunk_data(self, dataset: List[dict], text_content_key: str = 'context'):
    """
    Splits dataset entries into overlapping text chunks based on 'word' count.

    Args:
        dataset (list[dict]): List of data items
        text_content_key (str, optional): Key to extract text content from each data item. Defaults to 'context'.

    Returns:
        tuple:
            list[str]: List of text chunks.
            list[dict]: List of metadata dictionaries for each chunk, including:
                - 'document_id': ID of the source document.
                - 'data_chunk_id': Sequential chunk ID.
                - 'start_word_index': Starting word index in the source text.
    """
    data_chunks = []
    data_chunks_metadata = []
    for item in dataset:
      text = item.get(text_content_key, "")
      if not text:
          continue
      words_list = text.split()
      if len(words_list)<=self.max_data_chunk_len:
          data_chunks.append(text)
          data_chunks_metadata.append({'document_id': self._document_id_tracker, 'data_chunk_id': 0, 'start_word_index': 0})
          self._document_id_tracker += 1
          continue
      start_word_index = 0
      data_chunk_id = 0
      while start_word_index < len(words_list):
          chunk = words_list[start_word_index:start_word_index + self.max_data_chunk_len]
          data_chunks.append(" ".join(chunk))
          data_chunks_metadata.append({
              'document_id': self._document_id_tracker,
              'data_chunk_id': data_chunk_id,
              'start_word_index': start_word_index
          })
          if start_word_index + self.max_data_chunk_len >= len(words_list):
              break
          start_word_index += self.max_data_chunk_len - self.data_chunk_stride
          data_chunk_id += 1
      self._document_id_tracker += 1
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


  def add_update_data_and_index(self, dataset: List[dict], text_content_key: str = 'context'):
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
        text_content_key (str, optional): Key to extract text content from each data item. Defaults to 'context'.

    Returns:
        None
    """
    new_data_entries = self._find_new_data_entries(dataset, text_content_key)
    if not new_data_entries:
      return
    new_data_chunks, new_data_chunks_metadata = self._chunk_data(new_data_entries, text_content_key)
    new_embeddings = self._calculate_sentence_embeddings(new_data_chunks)
    self._create_update_faiss_index(new_embeddings)
    self.raw_dataset.extend(new_data_entries)
    self.chunked_data.extend(new_data_chunks)
    self.chunked_data_metadata.extend(new_data_chunks_metadata)