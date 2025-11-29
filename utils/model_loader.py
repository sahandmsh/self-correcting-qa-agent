
from base.constants import Constants
from typing import Dict, List, Union
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

class ModelLoader:
    def __init__(self):
        pass
  

    def load_hf_tokenizer(self, hugging_face_token: str, tokenizer_model: str = Constants.ModelNames.HuggingFace.CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2):
        """
        Loads a pretrained Hugging Face tokenizer.

        Args:
            hugging_face_token (str): Authentication token for accessing private or gated Hugging Face models.
            tokenizer_model (str, optional): Model name or path for the tokenizer.
                Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".

        Returns:
            AutoTokenizer: Pretrained tokenizer compatible with the cross-encoder model.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, token = hugging_face_token)
        return tokenizer


    def load_hf_cross_encoder(self, hugging_face_token: str, cross_encoder_model: str = Constants.ModelNames.HuggingFace.CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2):
        """
        Loads a pretrained cross-encoder model using sentence-transformers.

        Args:
            hugging_face_token (str): Authentication token for accessing private or gated Hugging Face models.
            cross_encoder_model (str, optional): Model name or path for the cross-encoder.
                Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".

        Returns:
            CrossEncoder: Cross-encoder model for scoring query–passage pairs.
        """
        cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)
        return cross_encoder


    def load_sentence_embedding_model(self, hugging_face_token: str, sentence_embedding_model: str = Constants.ModelNames.HuggingFace.SENTENCE_EMBEDDING_MINILM_L6_V2):
        """
        Loads a pretrained SentenceTransformer model from Hugging Face.

        Args:
            hugging_face_token (str): Authentication token for accessing private or gated Hugging Face models.
            sentence_embedding_model (str, optional): Model name or path for the SentenceTransformer.
                Defaults to "all-MiniLM-L6-v2".

        Returns:
            SentenceTransformer: Bi-encoder model for embedding text chunks or sentences.
        """
        sentence_transformer = SentenceTransformer(sentence_embedding_model, token = hugging_face_token)
        return sentence_transformer


    def load_gemini_generative_model(self, google_api_key: str, config, model_name: str = Constants.ModelNames.Gemini.GEMINI_2_5_FLASH):
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
    

    def load_hf_generative_model(self, hugging_face_token: str, generative_model_name: str = Constants.ModelNames.Qwen.QWEN_2_5_1_5B_INSTRUCT):
        """
        Loads a Hugging Face generative model for text generation.

        Args:
            hugging_face_token (str): Authentication token for accessing private or gated Hugging Face models.
            generative_model_name (str, optional): Model name or path for the generative model.
                Defaults to "Qwen/Qwen2.5-1.5B-Instruct".

        Returns:
            Callable: A function `generate(prompt: str)` that produces text
            for a given prompt using the specified model.
        """
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = self.load_hf_tokenizer(hugging_face_token, generative_model_name)
        model = AutoModelForCausalLM.from_pretrained(generative_model_name, token = hugging_face_token)
        model = model.to(device)
        
        def generate(prompt: str):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, max_new_tokens = 500, temperature = 0.7)
            generated_tokens = outputs[0][input_len:]
            return tokenizer.decode(generated_tokens, skip_special_tokens=True)
        


        

        def new_generate_to_be_used(
            prompt_or_messages: Union[str, List[Dict[str, str]]],
            max_new_tokens: int = 500,
            temperature: float = 0.7,
        ):
            """
            Generate text from a string prompt or chat-style messages.
            
            Args:
                prompt_or_messages: str or list of dicts with "role" and "content"
                max_new_tokens: number of tokens to generate
                temperature: sampling temperature
                device: torch device (e.g., "mps", "cuda", "cpu"). Defaults to model device.
                
            Returns:
                Generated text as string.
            """

            # Determine if input is chat messages or plain string
            if isinstance(prompt_or_messages, str):
                # plain string
                inputs = tokenizer(
                    prompt_or_messages,
                    return_tensors="pt"
                )
            elif isinstance(prompt_or_messages, list):
                # assume list of messages with {"role": ..., "content": ...}
                inputs = tokenizer.apply_chat_template(
                    prompt_or_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            else:
                raise ValueError("prompt_or_messages must be str or list of messages")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            generated_tokens = outputs[0][input_len:]
            return tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        
        
        
        
        
        return generate