
from google import genai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelLoader:
    def __init__(self):
        pass
  

    def load_hf_models(self, hugging_face_token: str, tokenizer_model: str ="cross-encoder/ms-marco-MiniLM-L-6-v2", cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", sentence_embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Loads pretrained tokenizer, cross-encoder, and sentence embedding models from Hugging Face.

        This utility function initializes and returns all major model components required for a
        Retrieval-Augmented Generation (RAG) pipeline, including:
        - A tokenizer for text encoding.
        - A cross-encoder for reranking retrieved passages.
        - A SentenceTransformer model for bi-encoder-style sentence or chunk embeddings.

        Args:
            hugging_face_token (str): Authentication token for accessing private or gated Hugging Face models.
            tokenizer_model (str, optional): Model name or path for the tokenizer.
                Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
            cross_encoder_model (str, optional): Model name or path for the cross-encoder.
                Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
            sentence_embedding_model (str, optional): Model name or path for the SentenceTransformer.
                Defaults to "all-MiniLM-L6-v2".

        Returns:
            tuple:
                tokenizer (AutoTokenizer): Pretrained tokenizer compatible with the cross-encoder model.
                cross_encoder (AutoModelForSequenceClassification): Cross-encoder model for scoring query–passage pairs.
                sentence_transformer (SentenceTransformer): Bi-encoder model for embedding text chunks or sentences.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token = hugging_face_token)
        cross_encoder = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model, token = hugging_face_token)
        sentence_transformer = SentenceTransformer(sentence_embedding_model, token = hugging_face_token)
        return tokenizer, cross_encoder, sentence_transformer


    def load_gemini_model(self, google_api_key: str, config, model_name: str = "gemini-2.5-flash"):
        """
        Loads a Gemini generative model for text generation.

        Args:
            google_api_key (str): API key for the Google Generative AI service.
            config (dict): Model configuration (e.g., temperature, max_output_tokens).
            model_name (str): Name of the generative model to load.

        Returns:
            Callable: A function `generate(prompt: str)` that produces text
            for a given prompt using the specified model.
        """
        client = genai.Client(api_key = google_api_key)
        def generate(prompt: str):
            response = client.models.generate_content(model = model_name, contents= prompt, config = config)
            return response.text
        return generate