# Agentic AI - Intelligent QA Agent with RAG and Web Search

An intelligent question-answering agent that combines Retrieval-Augmented Generation (RAG) with real-time web search capabilities. The agent automatically routes queries to the most appropriate tool—whether that's an indexed knowledge base, live web search, or the model's general knowledge.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Why This Project?

This repository demonstrates end-to-end open-loop agentic AI design:
- Custom tool routing with strict JSON schemas
- Modern RAG pipelines with re-ranking
- Real-time web search integration
- LLM orchestration

It serves both as a learning project and a portfolio demonstration of applied NLP, and agentic system design.

## 🌟 Features

- **Intelligent Tool Routing**: Automatically selects the best tool (RAG, web search, or general knowledge) based on query context
- **Advanced RAG Pipeline**: 
  - Bi-encoder (FAISS) for fast initial retrieval
  - Cross-encoder re-ranking for precision
  - Sliding-window chunking with overlap for context preservation
- **Real-time Web Search**: DuckDuckGo integration with content extraction via Trafilatura
- **Flexible LLM Support**: Integrated with Google Gemini (2.5 Pro/Flash) and Hugging Face models
- **Modular Architecture**: Clean separation of concerns with extensible tool framework

## 🏗️ Architecture

```
agentic_ai/
├── agent/                      # Core agent logic
│   ├── agent.py               # QAAgent with tool routing
│   └── tools/                 # Tool implementations
│       ├── rag_tool.py        # Knowledge base retrieval
│       └── web_search_tool.py # Live web search
├── rag/                       # RAG components
│   ├── rag_corpus_manager.py  # Document chunking & indexing
│   └── rag_content_retriever.py # Retrieval & re-ranking
├── utils/                     # Utilities
│   ├── model_loader.py        # Model initialization
│   ├── web_search.py          # Web search implementation
│   └── dataset_utils.py       # Dataset loading utilities
├── base/                      # Constants & configurations
│   └── constants.py           # System prompts & model names
├── config.py                  # Environment configuration
└── main.py                    # Entry point with examples
```

### Quick Start

```python
from agent.agent import QAAgent
from rag.rag_corpus_manager import RAGCorpusManager
from rag.rag_content_retriever import RAGContentRetriever
from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool
from utils.model_loader import ModelLoader
from config import HF_TOKEN, GOOGLE_API_KEY

# Initialize models
model_loader = ModelLoader()
sentence_transformer = model_loader.load_sentence_embedding_model(HF_TOKEN)
cross_encoder = model_loader.load_hf_cross_encoder(HF_TOKEN)
generative_model = model_loader.load_gemini_generative_model(GOOGLE_API_KEY, config)

# Set up RAG components
rag_corpus = RAGCorpusManager(sentence_transformer)
rag_retriever = RAGContentRetriever(cross_encoder, generative_model)

# Initialize tools
rag_tool = RagTool(rag_corpus, rag_retriever)
web_search_tool = WebSearchTool(rag_retriever, RAGCorpusManager(sentence_transformer))

# Create agent
agent = QAAgent(rag_tool, web_search_tool, generative_model)

# Ask questions
response = agent.chat("What is Agentic AI?")
print(response)
```

## 💡 Usage Examples

### 1. Knowledge Base Queries
```python
# Load your documents
knowledge_base = [
    {"context": "Notre Dame University was founded in 1842..."},
    {"context": "The current president is Rev. John Jenkins..."}
]
rag_corpus.add_update_data_and_index(knowledge_base)

# Query the knowledge base
query = "Based on the knowledge base, who is the president of Notre Dame?"
response, tool = agent.chat(query)
print(f"Answer: {response}")
print(f"Tool used: {tool}")  # Output: rag_tool
```

### 2. Web Search Queries
```python
# Real-time information
query = "What is the latest Gemini model?"
response, tool = agent.chat(query)
print(f"Answer: {response}")
print(f"Tool used: {tool}")  # Output: web_search_tool

# Current events
query = "What was Google's stock price on November 25, 2025?"
response, tool = agent.chat(query)
```

### 3. General Knowledge (No Tool)
```python
# Direct model knowledge
query = "Who was the first president of Notre Dame? Use your own knowledge."
response, tool = agent.chat(query)
print(f"Tool used: {tool}")  # Output: none
```

## 🔧 Configuration

### Model Selection

The system supports multiple models configured in `base/constants.py`:

```python
# Embedding models
SENTENCE_EMBEDDING_MINILM_L6_V2 = "all-MiniLM-L6-v2"
CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2 = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generative models
GEMINI_2_5_PRO = "gemini-2.5-pro"
GEMINI_2_5_FLASH = "gemini-2.5-flash"
QWEN_2_5_1_5B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
```

### RAG Parameters

Customize chunking and retrieval in `RAGCorpusManager`:

```python
rag_corpus = RAGCorpusManager(
    sentence_transformer=sentence_transformer,
    max_data_chunk_len=300,      # Words per chunk
    data_chunk_stride=75          # Overlap between chunks
)
```

Adjust retrieval precision:

```python
top_results = retriever.find_top_similar_items(
    rag_corpus,
    query,
    biencoder_top_k=100,          # Initial candidates
    cross_encoder_top_k=5,        # Final re-ranked results
    cross_encoder_batch_size=16
)
```

## 🧠 How It Works

### 1. Tool Routing
The agent uses a three-mode system prompt architecture:

- **Mode A (Tool Router)**: Analyzes queries to select appropriate tools
- **Mode B (Context-Preferred Answerer)**: Generates responses using retrieved context
- **Mode C (General Assistant)**: Answers from model's general knowledge

### 2. RAG Pipeline

```
Query → Bi-encoder (FAISS) → Top-K Candidates → Cross-encoder Re-ranking → 
Merge Overlapping Chunks → Generate Response
```

**Key Components:**
- **Chunking**: Sliding window (300 words, 75-word stride) preserves context
- **Deduplication**: Normalized text comparison prevents redundant indexing
- **Two-stage Retrieval**: Fast approximate search + precise re-ranking
- **Chunk Merging**: Reconstructs coherent passages from overlapping segments

### 3. Web Search Integration

⚠️ Disclaimer
This tool is provided for educational and research purposes only.
It does not enforce robots.txt restrictions.
Users are responsible for ensuring that their usage complies with 
the terms of service and robots.txt rules of any website they access.
The author assumes no responsibility for misuse.

```
Query → DuckDuckGo Search → Fetch Web Pages → Extract Content → 
Index in Temporary RAG Corpus → Retrieve & Generate
```

## 📊 Components

### QAAgent
Central coordinator managing tool selection and response generation.

**Key Methods:**
- `chat(query: str) -> Tuple[str, str]`: Main interface returning (response, tool_used)
- `_tool_router(query: str)`: Intelligent tool selection logic

### RAGCorpusManager
Handles document ingestion, chunking, embedding, and FAISS indexing.

**Key Features:**
- Automatic deduplication via normalized text comparison
- Sliding-window chunking for context preservation
- FAISS inner-product index for cosine similarity search

### RAGContentRetriever
Performs two-stage retrieval with optional response generation.

**Retrieval Pipeline:**
1. Bi-encoder FAISS search (fast, approximate)
2. Cross-encoder re-ranking (slow, precise)
3. Chunk merging for coherent passages
4. Optional LLM-based answer generation

### WebSearchTool
Executes real-time web searches and applies RAG over results.

**Features:**
- DuckDuckGo search backend
- Trafilatura content extraction
- Ephemeral RAG corpus (cleared per query)

## 📝 System Prompts

The agent uses carefully engineered system prompts for:

1. **Tool Selection**: JSON-only output with strict formatting rules
2. **Context-based Answering**: Prefers retrieved context while allowing fallback to general knowledge
3. **General Assistance**: Direct, concise responses

Prompts are modular and defined in `base/constants.py` for easy customization.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for more LLM providers (OpenAI, Anthropic, etc.)
- [ ] Implement conversation history/memory
- [ ] Add query result caching
- [ ] Support for multi-modal inputs (images, PDFs)
- [ ] Evaluation metrics and benchmarks
- [ ] Async processing for improved performance
- [ ] Self-correcting agent for better responses and tool selection

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Sentence-Transformers](https://www.sbert.net/) for semantic embeddings
- Vector indexing powered by [FAISS](https://github.com/facebookresearch/faiss) (Meta AI)
- LLM inference via [Google Gemini](https://deepmind.google/technologies/gemini/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- Cross-encoder re-ranking from [Sentence-Transformers](https://www.sbert.net/)
- Web search via [DuckDuckGo](https://duckduckgo.com/)
- Content extraction by [Trafilatura](https://trafilatura.readthedocs.io/)

## 📧 Contact

**Sahand Mosharafian**
- GitHub: [@sahandmsh](https://github.com/sahandmsh)
- Project Link: [https://github.com/sahandmsh/agentic_ai](https://github.com/sahandmsh/agentic_ai)

## 🔖 Citation

If you use this project in your research, please cite:

```bibtex
@software{mosharafian2025agenticai,
  author = {Mosharafian, Sahand},
  title = {Agentic AI: Intelligent QA Agent with RAG and Web Search},
  year = {2025},
  url = {https://github.com/sahandmsh/agentic_ai}
}
```

---

⭐ **Star this repository if you find it helpful!**
