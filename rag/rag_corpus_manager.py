from sentence_transformers import SentenceTransformer
from typing import List, Tuple

import faiss
import numpy as np


class RAGCorpusManager:
    """Manage corpus ingestion, chunking, embedding, and indexing for RAG.

    Responsibilities:
        * Text normalization & deduplication to avoid redundant storage/embedding.
        * Sliding-window chunking with configurable length and stride.
        * Embedding computation via a provided `SentenceTransformer`.
        * FAISS (inner product) index creation/update for fast similarity search.

    Attributes:
        raw_dataset (list[dict]): Original unique document entries.
        chunked_data (list[str]): Generated text chunks.
        chunked_data_metadata (list[dict]): Metadata per chunk (document_id, data_chunk_id, start_word_index).
        sentence_embeddings (list[np.ndarray] or np.ndarray): Stored embeddings (if retained externally).
        sentence_transformer (SentenceTransformer): Model used for embedding generation.
        max_data_chunk_len (int): Maximum words per chunk.
        data_chunk_stride (int): Overlap stride between successive chunks.
    """

    def __init__(
        self,
        sentence_transformer: SentenceTransformer,
        max_data_chunk_len: int = 300,
        data_chunk_stride: int = 75,
    ):
        """Initialize the corpus manager.

        Args:
            sentence_transformer (SentenceTransformer): Preloaded sentence transformer used to encode text.
            max_data_chunk_len (int, optional): Maximum number of words per chunk. Defaults to 300.
            data_chunk_stride (int, optional): Overlap (in words) between consecutive chunks. Defaults to 75.

        Raises:
            ValueError: If `max_data_chunk_len` <= 0 or if stride is invalid (negative or >= max length).
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
            raise ValueError(
                "Invalid stride value; make sure that 0<= data_chunk_stride < max_data_chunk_len"
            )

    def _clear_corpus(self) -> None:
        """
        Clears the existing corpus data and resets internal states.

        This method removes all stored documents, chunked data, embeddings,
        and resets the document ID tracker and unique text set. Should be used
        when starting a fresh corpus ingestion session only.
        """
        self._document_id_tracker = 0
        self.raw_dataset = []
        self._unique_text_set = set()
        self.chunked_data = []
        self.chunked_data_metadata = []
        self.sentence_embeddings = []
        if hasattr(self, "faiss_index"):
            del self.faiss_index

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes input text by trimming extra spaces and line breaks.

        Args:
            text (str): Input text string.

        Returns:
            str: Normalized text with single spaces between words.
        """
        return " ".join(text.strip().split())

    def _find_new_data_entries(
        self, new_dataset: List[dict], text_content_key: str = "context"
    ) -> List[dict]:
        """Filter out duplicates from an incoming dataset.

        Normalizes text and checks against an internal set of seen strings.

        Args:
            new_dataset (list[dict]): Candidate data entries.
            text_content_key (str, optional): Key containing text content. Defaults to "context".

        Returns:
            list[dict]: Unique entries whose normalized text was not previously stored.
        """
        new_unique_dataset = []
        for item in new_dataset:
            normalized_text = self._normalize_text(item.get(text_content_key, ""))
            if normalized_text and normalized_text not in self._unique_text_set:
                self._unique_text_set.add(normalized_text)
                new_unique_dataset.append(item)
        return new_unique_dataset

    def _chunk_data(
        self, dataset: List[dict], text_content_key: str = "context"
    ) -> Tuple[List[str], List[dict]]:
        """Chunk each document into overlapping word windows.

        Applies a sliding window of size `max_data_chunk_len` with stride
        `max_data_chunk_len - data_chunk_stride` until the end of the word list.

        Args:
            dataset (list[dict]): Data items containing raw text.
            text_content_key (str, optional): Key holding text content. Defaults to "context".

        Returns:
            tuple[list[str], list[dict]]: Text chunks and their metadata dicts:
                document_id, data_chunk_id, start_word_index.
        """
        data_chunks = []
        data_chunks_metadata = []
        for item in dataset:
            text = item.get(text_content_key, "")
            if not text:
                continue
            words_list = text.split()
            if len(words_list) <= self.max_data_chunk_len:
                data_chunks.append(text)
                data_chunks_metadata.append(
                    {
                        "document_id": self._document_id_tracker,
                        "data_chunk_id": 0,
                        "start_word_index": 0,
                    }
                )
                self._document_id_tracker += 1
                continue
            start_word_index = 0
            data_chunk_id = 0
            while start_word_index < len(words_list):
                chunk = words_list[start_word_index : start_word_index + self.max_data_chunk_len]
                data_chunks.append(" ".join(chunk))
                data_chunks_metadata.append(
                    {
                        "document_id": self._document_id_tracker,
                        "data_chunk_id": data_chunk_id,
                        "start_word_index": start_word_index,
                    }
                )
                if start_word_index + self.max_data_chunk_len >= len(words_list):
                    break
                start_word_index += self.max_data_chunk_len - self.data_chunk_stride
                data_chunk_id += 1
            self._document_id_tracker += 1
        return data_chunks, data_chunks_metadata

    def _calculate_sentence_embeddings(
        self, data_chunks: List[str], batch_size: int = 64
    ) -> np.ndarray:
        """Compute embeddings for provided text chunks.

        Args:
            data_chunks (list[str]): Text chunks to encode.
            batch_size (int, optional): Batch size for model inference. Defaults to 64.

        Returns:
            np.ndarray: Embedding matrix shape (num_chunks, embedding_dim).
        """
        sentence_embeddings = self.sentence_transformer.encode(
            data_chunks, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return sentence_embeddings

    def _create_update_faiss_index(self, embeddings: np.ndarray):
        """Create (if needed) and append embeddings to a cosine-similarity FAISS index.

        Embeddings are L2-normalized before insertion. If no index exists a new
        `IndexFlatIP` is created using the embedding dimensionality.

        Args:
            embeddings (np.ndarray): Chunk embeddings to add.
        """
        faiss.normalize_L2(embeddings)
        if not hasattr(self, "faiss_index"):
            embedding_dim = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(embeddings)

    def add_update_data_and_index(self, dataset: List[dict], text_content_key: str = "context"):
        """Ingest new dataset entries, embedding and indexing any unique content.

        Pipeline:
          1. Filter duplicates.
          2. Chunk new texts.
          3. Compute embeddings.
          4. Update FAISS index.
          5. Extend internal storage structures.

        Args:
            dataset (list[dict]): Incoming data entries.
            text_content_key (str, optional): Key for text content. Defaults to "context".

        Returns:
            None
        """
        new_data_entries = self._find_new_data_entries(dataset, text_content_key)
        if not new_data_entries:
            return
        new_data_chunks, new_data_chunks_metadata = self._chunk_data(
            new_data_entries, text_content_key
        )
        new_embeddings = self._calculate_sentence_embeddings(new_data_chunks)
        self._create_update_faiss_index(new_embeddings)
        self.raw_dataset.extend(new_data_entries)
        self.chunked_data.extend(new_data_chunks)
        self.chunked_data_metadata.extend(new_data_chunks_metadata)
