from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool
from base.constants import Constants
from typing import Callable, Dict, Any, Tuple

import json


class QAAgent:
    """Open-loop conversational agent equipped with retrieval tool, web search tool,and a generative model.

    Attributes:
        generative_model (Callable[[str], str]): Function or model interface used to generate
            natural language responses.
        tools (Dict[str, Dict[str, Any]]): Registry containing tool instances and metadata.
    """

    def __init__(
        self,
        rag_tool: RagTool,
        web_search_tool: WebSearchTool,
        generative_model: Callable[[str], str],
    ) -> None:
        """Initialize the QAAgent.

        Args:
            rag_tool (RagTool): Internal knowledge-base retrieval & generation wrapper.
            web_search_tool (WebSearchTool): Live web search + retrieval interface.
            generative_model (Callable[[str], str]): Callable that accepts a prompt and returns a
                generated string response.
        """
        self.generative_model = generative_model
        self.tools: Dict[str, Dict[str, Any]] = {
            "rag_tool": {
                "instance": rag_tool,
                "description": (
                    "Searches the indexed knowledge base for answers. Use this for questions "
                    "about specific documents, facts in the knowledge base, or structured "
                    "information that has been previously indexed. Best for factual queries "
                    "where the answer should come from the internal knowledge base."
                ),
                "args": {
                    "query": "The question whose answer should be drawn from indexed knowledge.",
                    "max_results": "Maximum number of top similar items to retrieve.",
                },
            },
            "web_search_tool": {
                "instance": web_search_tool,
                "description": (
                    "Performs real-time internet search for current, recent, or trending "
                    "information not in the knowledge base. Use for up-to-date news or events."
                ),
                "args": {
                    "query": "The question to search on the web.",
                    "max_web_pages": "Maximum number of web search results to fetch.",
                    "max_top_passages": "Number of top passages to retrieve from RAG.",
                },
            },
        }

    def _tool_router(self, query: str) -> Dict[str, Any]:
        """Select a tool by prompting the generative model with a routing instruction.

        The model is expected to return JSON like: {"tool": "rag_tool"} or {"tool": "web_search_tool"} or {"tool": "none"}.
        Heuristics attempt JSON recovery if the raw output is not strictly valid.

        Args:
            query (str): User query text.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary containing at least the key 'tool'.
        """
        prompt = Constants.Instructions.System.SystemModesTags.TOOL_ROUTER + "\n" + query
        response = str(self.generative_model(prompt))
        try:
            return json.loads(response)
        except Exception:
            pass
        try:
            start = response.index("{")
            end = response.rindex("}")
            snippet = response[start : end + 1]
            return json.loads(snippet)
        except:
            return {"tool": "none"}

    def chat(self, query: str) -> Tuple[str, str]:
        """Process a user query, routing through an appropriate tool and generating a reply.

        Args:
            query (str): User query.

        Returns:
            Tuple[str, str]: (agent_response, tool_used)
                agent_response: Final generated answer.
                tool_used: Name of tool selected or 'none'.
        """
        tool_decision = self._tool_router(query)
        tool_name = tool_decision.get("tool", "none")
        if tool_name == "none":
            prompt = Constants.Instructions.System.SystemModesTags.GENERAL_ASSISTANT + "\n" + query
            response = self.generative_model(prompt=prompt)
        else:
            tool = self.tools[tool_name]["instance"]
            args = tool_decision.get("args", {})
            context = tool.use_tool(**args)
            prompt = (
                Constants.Instructions.System.SystemModesTags.CONTEXT_PREFERRED_ANSWERER
                + "\nContext: "
                + str(context)
                + "\nQuery: "
                + query
            )
            response = self.generative_model(prompt)
        return response, tool_name
