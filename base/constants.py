class Constants:

    class Timeout:
        ONE_SECOND = 1
        THREE_SECONDS = 3
        FIVE_SECONDS = 5
        TEN_SECONDS = 10
        THIRTY_SECONDS = 30
        ONE_MINUTE = 60

    class HTTPStatusCodes:
        OK = 200

    
    class URLHeader:
        USER_AGENT_HEADER = "Mozilla/5.0 (compatible; AI-Research-Bot/1.0; +https://github.com/sahandmsh/AgenticAI)"


    class WebSearch:
        DUCK_DUCK_GO_API_URL = "https://api.duckduckgo.com/"
    

    class ToolNames:
        WEB_SEARCH = "web_search"
        RAG_TOOL = "rag_tool"

    class ModelNames:

        class HuggingFace:
            CROSS_ENCODER_MS_MARCO_MINILM_L_6_V2 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
            SENTENCE_EMBEDDING_MINILM_L6_V2 = "all-MiniLM-L6-v2"
        
        class Gemini:
            GEMINI_2_5_PRO = "gemini-2.5-pro"
            GEMINI_2_5_FLASH = "gemini-2.5-flash"
        
        class Qwen:
            QWEN_2_5_1_5B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"

    class Instructions:

        RAG_CONTEXT_BASED_INSTRUCTION = """
            You are an AI assistant that answers ONLY using the information provided in the context.

            RULES:
            - If the answer is not explicitly stated in the context, respond with: "I don't know."
            - Do NOT use any outside knowledge.
            - Do NOT guess or infer beyond what the context states.
            - Stay on topic.
            - Keep answers short and directly address the query.

            Query: {query}
            Context: {context}
        """
        
        CONTEXT_BASED_INSTRUCTION = """
            Here is the context: {context}
            Use the provided context to answer the following question: {query}

            If you cannot find the answer to the question in the context, let the user know.
            Then, try to answer the question based on your own knowledge.
            Stay concise and to the point. Do not go off topic.
        """
        QA_INSTRUCTION = """
            Stay concise and to the point. Do not go off topic. Keep the answer brief.
            If you do not know the answer, say 'I don't know'.
            Provide answer to the user's query: {query}
        """
        TOOL_SELECTION_INSTRUCTION = """
            You are a tool router, not a question-answering assistant. 
            Your ONLY job is to decide whether the user's query requires a tool. 
            If no tool is needed, choose "none".\n

            THINK CAREFULLY but do NOT show your reasoning. 
            Output ONLY the final JSON object.\n

            RULES:
            - "web_search_tool": use ONLY for recent, time-sensitive information or anything that clearly requires a web search.
            - "rag_tool": use ONLY when the answer must come from the indexed knowledge base 
                (e.g., questions about specific documents, files, stored passages, or private data).
            Do NOT use rag_tool for general knowledge, common facts, or anything the model can answer directly.
            - "none": choose this for ALL other cases. This is the DEFAULT option unless the query clearly requires a tool.
            - When unsure, choose "none".

            OUTPUT FORMAT:
            Return exactly ONE of these JSON structures:
            {{"tool": "rag_tool", "args": {{"query": "..."}}}}
            {{"tool": "web_search_tool", "args": {{"query": "..."}}}}
            {{"tool": "none"}}

            Do NOT output explanations, reasoning, or commentary. 
            Do NOT wrap the JSON in code fences.
            Only return ONE valid JSON object as specified.

            EXAMPLES:
            User: "Explain how transformers work."
            → {{"tool": "none"}}

            User: "According to our uploaded documents, what is the warranty duration?"
            → {{"tool": "rag_tool", "args": {{"query": "warranty duration"}}}}

            User: "What were the stock prices yesterday?"
            → {{"tool": "web_search_tool", "args": {{"query": "stock prices yesterday"}}}}

            Now process this:
            User Query: {query}
        """