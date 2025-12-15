# Self-correcting QA Agent:

An intelligent question-answering agent that combines Retrieval-Augmented Generation (RAG) with real-time web search capabilities using a **ReAct (Reasoning and Acting) framework** with self-correction. The agent automatically routes queries to the most appropriate tool—whether that's an indexed knowledge base, live web search, or the model's general knowledge—and iteratively refines its responses through self-reflection and feedback loops.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Why This Project?

This repository demonstrates end-to-end ReAct agentic AI design with self-correction:
- **ReAct Framework**: Iterative reasoning-action-reflection loops
- **Self-Correction**: Autonomous reflection and response refinement
- Custom tool routing with strict JSON schemas
- Modern RAG pipelines with re-ranking
- Real-time web search integration
- Session memory and query-level memory management

It serves both as a learning project and a portfolio demonstration of advanced agentic systems, applied NLP, and autonomous AI design.

## 🌟 Features

### Core Agent Capabilities
- **ReAct Architecture**: Implements the Reasoning-Acting-Reflection loop for autonomous problem-solving
- **Self-Correction Loop**: Agent reflects on its responses and iteratively improves them (up to configurable max iterations)
- **Multi-Mode Operation**: 6 operational modes (Tool Router, Context-Based, General Assistant, Self-Observation, Follow-Up Decision, Web Search)
- **Session & Query Memory**: Maintains conversation context and tracks iterative improvement attempts
- **Intelligent Tool Routing**: Automatically selects the best tool (RAG, web search, or general knowledge) based on query context

### Retrieval & Knowledge Management
- **Advanced RAG Pipeline**: 
  - Bi-encoder (FAISS) for fast initial retrieval
  - Cross-encoder re-ranking for precision
  - Sliding-window chunking with overlap for context preservation
- **Real-time Web Search**: DuckDuckGo integration with content extraction via Trafilatura
- **Persistent History**: JSON-based conversation history with structured logging

### Technical Infrastructure
- **Flexible LLM Support**: Integrated with Google Gemini (2.5 Pro/Flash) and Hugging Face models
- **Modular Architecture**: Clean separation of concerns with extensible tool framework
- **Error Handling**: Robust exception management with detailed error tracking

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

# Create agent (with optional history file and max iterations)
agent = QAAgent(
    rag_tool, 
    web_search_tool, 
    generative_model,
    history_file_path="agent_history.json",
    max_history_entries=100
)

# Ask questions - agent will self-correct through ReAct loop
response, context = agent.agent_chat("What is Agentic AI?", max_tries=5)
print(response)
```

## ReAct Loop Example

Here's what happens when you call `agent.agent_chat()`:

```python
response, context = agent.agent_chat("What is the capital of France?", max_tries=5)
```

**Behind the scenes:**

1. **Iteration 1:**
   - Tool Router: Decides tool="none" (general knowledge)
   - Response: "Paris is the capital of France."
   - Reflection: "Response is accurate and complete."
   - Follow-up: `<affirm>` → ✓ Done!

For more complex queries requiring tool use:

```python
response, context = agent.agent_chat("What are the latest updates to Gemini models?", max_tries=5)
```

**Behind the scenes:**

1. **Iteration 1:**
   - Tool Router: Decides tool="web_search_tool"
   - Web Search: Fetches 5 web pages, retrieves top passages
   - Response: Generated from web content
   - Reflection: "Information seems incomplete, only covered basic features"
   - Follow-up: `<revise>` with guidance to search for more specific details

2. **Iteration 2:**
   - Tool Router: Uses web_search_tool again with refined query
   - Web Search: Fetches updated content
   - Response: More comprehensive answer
   - Reflection: "Response now addresses all aspects of the query"
   - Follow-up: `<affirm>` → ✓ Done!

The agent autonomously identifies gaps and refines its approach without human intervention.

## �💡 Usage Examples

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
response, context = agent.agent_chat(query)
print(f"Answer: {response}")
# Agent will use rag_tool and self-correct if needed
```

### 2. Web Search Queries
```python
# Real-time information
query = "What is the latest Gemini model?"
response, context = agent.agent_chat(query)
print(f"Answer: {response}")
# Agent will use web_search_tool and iterate until satisfied
```

### 3. General Knowledge (No Tool)
```python
# Direct model knowledge
query = "What is machine learning?"
response, context = agent.agent_chat(query)
# Agent will decide no tool is needed and answer directly
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

### ReAct Loop Configuration

Control the self-correction behavior:

```python
# Conservative: Quick answers with minimal iteration
response, context = agent.agent_chat(query, max_tries=2)

# Balanced: Default setting for most queries
response, context = agent.agent_chat(query, max_tries=5)

# Thorough: Maximum effort for complex queries
response, context = agent.agent_chat(query, max_tries=10)
```

**Note**: The agent will stop early if it reaches `<affirm>` or `<failed>` state before max_tries.

## 🧠 How It Works

### 1. ReAct Loop with Self-Correction

The agent follows an iterative **Reasoning-Acting-Reflection** cycle:

```
Query → [Tool Routing → Tool Execution → Response Generation → 
        Self-Reflection → Follow-Up Decision] → Final Response
                              ↓ <revise>
                              ← (loop back with feedback)
```

**Key Steps:**
1. **Reasoning (Tool Router)**: Analyzes query and selects appropriate tool
2. **Acting**: Executes tool or generates direct response
3. **Reflection**: Self-evaluates response quality (reflection, mistake, guidance)
4. **Follow-Up Decision**: Decides to `<affirm>`, `<revise>`, or `<failed>`
5. **Iteration**: If `<revise>`, loops back with self-instructions (up to max_tries)

### 2. Six-Mode System Architecture

The agent operates in distinct modes via system prompts:

- **Mode A (Tool Router)**: Analyzes queries to select appropriate tools (JSON output)
- **Mode B (Context-Based)**: Generates responses strictly from retrieved context
- **Mode C (General Assistant)**: Answers from model's general knowledge
- **Mode D (Self-Observation)**: Reflects on actions and identifies improvements
- **Mode E (Follow-Up Decision)**: Decides if response needs revision
- **Mode F (Web Search)**: Synthesizes information from web search results

### 3. Memory Management

**Session Memory**: Tracks all interactions within a conversation session
**Query Memory**: Stores iteration attempts for the current query (used in self-correction loop)
**Persistent History**: JSON file with structured entries (timestamp, query, response, reflection, errors, etc.)

### 4. RAG Pipeline

```
Query → Bi-encoder (FAISS) → Top-K Candidates → Cross-encoder Re-ranking → 
Merge Overlapping Chunks → Generate Response
```

**Key Components:**
- **Chunking**: Sliding window (300 words, 75-word stride) preserves context
- **Deduplication**: Normalized text comparison prevents redundant indexing
- **Two-stage Retrieval**: Fast approximate search + precise re-ranking
- **Chunk Merging**: Reconstructs coherent passages from overlapping segments

### 5. Web Search Integration

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
Central coordinator implementing the ReAct loop with self-correction.

**Key Methods:**
- `agent_chat(query: str, max_tries: int = 5) -> Tuple[str, str]`: Main interface with ReAct loop returning (response, context)
- `_query_response(query: str, agent_feedback: str) -> Tuple[str, Dict, str, str]`: Executes one reasoning-acting cycle
- `_reflect(...)`: Self-reflection on agent's action (returns reflection, mistake, guidance)
- `_agent_feedback(...)`: Decides next action (<affirm>, <revise>, or <failed>)
- `_tool_router(query: str)`: Intelligent tool selection logic
- `_update_memory_and_history(...)`: Persists conversation state

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

The agent uses system prompts for all six modes:

1. **Tool Selection (Mode A)**: JSON-only output with strict formatting rules
2. **Context-Based Answering (Mode B)**: Strict adherence to retrieved context
3. **General Assistance (Mode C)**: Direct, concise responses from model knowledge
4. **Self-Observation (Mode D)**: Structured reflection on performance (JSON output)
5. **Follow-Up Decision (Mode E)**: Decision logic for iteration control
6. **Web Search Answering (Mode F)**: Synthesis and interpretation of web results

All prompts include:
- Global time awareness rules
- Brevity and conciseness guidelines
- Clear output formatting requirements
- Mode-specific behavior constraints

Prompts are modular and defined in `base/constants.py` for easy customization.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Enhanced memory compression for long conversations
- [ ] Query result caching to reduce redundant tool calls
- [ ] Multi-tool execution in single iteration
- [ ] Additional tools (calculator, code execution) for enhanced capabilities
- [ ] Adaptive max_tries based on query complexity

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
- Project Link: [https://github.com/sahandmsh/agentic-ai-2](https://github.com/sahandmsh/self-correcting-qa-agent)

## 🔖 Citation

If you use this project in your research, please cite:

```bibtex
@software{mosharafian2025selfcorrectingqaagent,
  author = {Mosharafian, Sahand},
  title = {Agentic AI: Self-Correcting ReAct Agent with RAG and Web Search},
  year = {2025},
  url = {https://github.com/sahandmsh/self-correcting-qa-agent}
}
```

---

⭐ **Star this repository if you find it helpful!**
