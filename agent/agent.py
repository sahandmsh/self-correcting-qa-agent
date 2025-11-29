from base.constants import Constants
from typing import List
from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool

import json
import os
import textwrap

# do some prompt engineering; maybe 2 step reasoning (thin/reason -> extract tool from the response by the LLM!). See if that works
# use mps for the rest of models in this file if possible
# Create a constants/configuration.json file to store settings
class QAAgent:
    def __init__(self, rag_tool: RagTool, web_search_tool: WebSearchTool, generative_model, memory_file: str = "memory.json"):
        """"""
        self.generative_model = generative_model
        self.tools = {
            "rag_tool": {
                "instance": rag_tool,
                "description": "Searches the indexed knowledge base for answers. Use this for questions about specific documents, facts in the knowledge base, or structured information that has been previously indexed. Best for factual queries where the answer should come from the internal knowledge base.",
                "args": {"query": "The question to find the answer for from the knowledge base."},
                "Returns": "The answer to the query from the knowledge base."
            },
            "web_search_tool": {
                "instance": web_search_tool,
                "description": "Performs real-time internet search for current, recent, or trending information not in the knowledge base. Use this for up-to-date news, latest developments, recent events, or information that requires real-time web access.",
                "args": {"query": "The question to search on the web."},
                "Returns": "The answer to the query from the web search."
            }
        }
        self.chat_history = []
        self.memory_file = memory_file
        # self._load_memory()


    def _load_memory(self):
        """ Loads chat history from the memory file if it exists.
        Returns:
            None
        Raises:
            FileNotFoundError: If the memory file does not exist.
        """
        try:
            with open(self.memory_file, "r") as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Could not find memory file. A new chat thread will be started.")


    def _save_memory(self):
        """
        Saves the current chat history to the memory file.
        Returns:
            None
        """
        with open(self.memory_file, "w") as f:
            json.dump(self.chat_history, f, indent = 4)


    def _choose_tool(self, query: str, instructions: str = Constants.Instructions.TOOL_SELECTION_INSTRUCTION):
        # LATER ADD HISTORY TO THE PROMPT
        # Short, focused prompt to avoid information overload
        # Use textwrap.dedent and .format to avoid f-string brace escaping issues
        prompt = instructions.format(query = query)
        response = str(self.generative_model(prompt))
        print(response)
        # Try direct JSON parse
        try:
            parsed = json.loads(response)
            return parsed
        except Exception:
            pass

        # Fallback: extract first balanced {...} substring and parse
        try:
            start = response.index('{')
            end = response.rindex('}')
            snippet = response[start:end+1]
            parsed = json.loads(snippet)
            return parsed
        except Exception:
            # final fallback: give up and return none
            return {"tool": "none"}
        

    def chat(
            self, 
            query: str, 
            context_based_instruction: str = Constants.Instructions.CONTEXT_BASED_INSTRUCTION, 
            qa_instruction: str = Constants.Instructions.QA_INSTRUCTION
        ):
        # Incomplete; will test first then complete iteratively
        """ Chats with the agent using the appropriate tool based on the query.
        Args:
            query (str): The user's query.
        Returns:
            str: The agent's response.
        """
        tool_decision = self._choose_tool(query)
        tool_name = tool_decision.get("tool", "none")
        if tool_name == "none":
            response = self.generative_model(qa_instruction.format(query = query))
        else:
            tool = self.tools[tool_name]['instance']


            # FIX FOLLOWING LATER SO IT WOULD BE ABLE TO PASS ARGS DYNAMICALLY
            context = tool.use_tool(query)
            prompt = context_based_instruction.format(context = context, query = query)
            response = self.generative_model(prompt)
        return response, tool_name
        