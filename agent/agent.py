from agent.tools.rag_tool import RagTool
from agent.tools.web_search_tool import WebSearchTool
from base.constants import Constants
from typing import Callable, Dict, Any, Tuple

import datetime
import json
import os


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
        history_file_path: str = "agent_history.json",
        max_history_entries: int = 100,
    ) -> None:
        """Initialize the QAAgent.

        Args:
            rag_tool (RagTool): Internal knowledge-base retrieval & generation wrapper.
            web_search_tool (WebSearchTool): Live web search + retrieval interface.
            generative_model (Callable[[str], str]): Callable that accepts a prompt and returns a
                generated string response.
            max_history_entries (int): Maximum number of recent history entries to pass to the model for context.
        """
        self.generative_model = generative_model
        self.session_memory = []
        self.query_memory = []
        self._history = self._load_history(history_file_path, max_history_entries)
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

    def _update_history_file(self, history_file_path: str, entry: Dict[str, str]) -> None:
        """Update the history file with a new conversation entry.
        Saves history as a JSON array of structured entries for better organization,
        queryability, and efficient access. Each entry contains timestamp, query,
        response, and reflection as separate fields.

        Args:
            history_file_path (str): Path to the history file.
            entry (Dict[str, str]): Dictionary containing 'timestamp', 'query',
                'response', 'reflection', and optionally 'tool_decision', 'context', 'errors'.
        """
        if not os.path.exists(history_file_path):
            with open(history_file_path, "w") as f:
                json.dump({"entries": [], "created_at": datetime.datetime.now().isoformat()}, f)
        with open(history_file_path, "r") as f:
            history = json.load(f)
        if "entries" not in history:
            history["entries"] = []
        history["entries"].append(entry)
        with open(history_file_path, "w") as f:
            json.dump(history, f, indent=4)

    def _load_history(self, history_file_path: str, max_history_entries: int = 100) -> list:
        """Load structured history entries and return the most recent ones up to max_history_entries.

        Args:
            history_file_path (str): Path to the history file.
            max_history_entries (int): Maximum number of recent history entries to load.

        Returns:
            list: history content.
        """
        if not os.path.exists(history_file_path):
            return []

        with open(history_file_path, "r") as f:
            history = json.load(f)
        return history["entries"][-max_history_entries:]

    def _tool_router(self, query: str) -> Dict[str, Any]:
        """Select a tool by prompting the generative model with a routing instruction.

        The model is expected to return JSON like: {"tool": "rag_tool"} or {"tool": "web_search_tool"} or {"tool": "none"}.
        Heuristics attempt JSON recovery if the raw output is not strictly valid.

        Args:
            query (str): User query text.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary containing at least the key 'tool'.
        """
        prompt = Constants.Instructions.AgenticAI.SystemModesTags.TOOL_ROUTER + "\n" + query
        response = str(self.generative_model(prompt))
        return json.loads(response)

    def _query_response(
        self, query: str, agent_feedback: str = ""
    ) -> Tuple[str, Dict[str, Any], str, str]:
        """Process a user query, routing through an appropriate tool and generating a reply.

        Args:
            query (str): User query.
            agent_feedback (str): Instructions for revising the response.
        Returns:
            Tuple[str, Dict[str, Any], str, str]: (response, tool_decision, context, errors)
                response: Final generated answer.
                tool_decision: Dictionary containing tool name and arguments.
                context: Context retrieved from the tool (if any).
                errors: Any errors encountered during processing.
        """
        errors = ""
        context = ""
        prompt = ""
        query_memory_str = self._format_session_memory(self.query_memory)
        try:
            tool_decision = self._tool_router(
                f"Query memory:\n{query_memory_str}"
                + f"\nquery: {query}\n"
                + f"IMPORTANT INSTRUCTIONS and self-feedback: {agent_feedback}"
            )
        except Exception as e:
            errors += f"Tool selection failed because of the following exception: {str(e)}"
            tool_decision = {"tool": "none"}
        tool_name = tool_decision.get("tool", "none")

        if tool_name == "none":
            mode_tag = Constants.Instructions.AgenticAI.SystemModesTags.GENERAL_ASSISTANT
        elif tool_name == "rag_tool":
            mode_tag = Constants.Instructions.AgenticAI.SystemModesTags.CONTEXT_BASED
        else:
            mode_tag = Constants.Instructions.AgenticAI.SystemModesTags.WEB_SEARCH

        if tool_name != "none":
            try:
                tool = self.tools[tool_name]["instance"]
                args = tool_decision.get("args", {})
                context = tool.use_tool(**args)
            except Exception as e:
                errors += f"Failed to use tool because of the following exception: {str(e)}"
        query_memory_str = self._format_session_memory(self.query_memory)
        prompt = (
            mode_tag
            + (f"Session memory: {query_memory_str}" if query_memory_str else "")
            + ("\nContext: " + str(context) if context else "")
            + ("\nWeb search results: " + str(context) if context else "")
            + f"\nTool used: {tool_decision}"
            + ("\nErrors when using tool: " + str(errors) if errors else "")
            + f"\nQuery: {query}"
            + (f"\nIMPORTANT INSTRUCTIONS: {agent_feedback}" if agent_feedback else "")
        )
        response = self.generative_model(prompt=prompt)
        return response, tool_decision, context, errors

    def _reflect(
        self,
        query: str,
        tool_decision: Dict[str, Any],
        response: str,
        errors: str = "",
        agent_feedback: str = "",
    ) -> Tuple[str, str, str]:
        """Generate a self-reflection on the agent's last action.

        Args:
            query (str): Original user query.
            tool_decision (Dict[str, Any]): Tool decision dictionary.
            response (str): Agent's generated response.
            errors (str): Any errors encountered during processing.
            agent_feedback (str): Instructions for revising the response.
        Returns:
            Tuple[str, str, str]: Agent's self-reflection components (reflection, mistake, guidance).
        """
        prompt = (
            Constants.Instructions.AgenticAI.SystemModesTags.SELF_OBSERVATION
            + "\nUser Query: "
            + str(query)
            + "\nTool Decision: "
            + str(tool_decision)
            + "\nAgent Response: "
            + str(response)
            + "\nAgent's Instructions: "
            + str(agent_feedback)
            + (f"\nErrors Encountered: {str(errors)}" if errors else "")
            + "\nProvide your self-reflection following the specified structure."
        )
        reflection_response = self.generative_model(prompt=prompt)
        try:
            # Strip markdown code blocks if present (e.g., ```json\n...\n```)
            cleaned = reflection_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            reflection = json.loads(cleaned)
            return (
                reflection.get("reflection", ""),
                reflection.get("mistake", ""),
                reflection.get("guidance", ""),
            )
        except Exception as e:
            return (
                reflection_response,
                f"JSON parsing error in reflection generation: {str(e)}",
                "Ensure the model returns valid JSON format.",
            )

    def _agent_feedback(
        self,
        query: str,
        reflection: str,
        mistake: str,
        guidance: str,
        previous_feedback: str,
    ) -> str:
        """
        Decide on the next action based on self-reflection.
        Args:
            query (str): Original user query.
            reflection (str): Agent's self-reflection.
            mistake (str): Identified mistakes in the previous response.
            guidance (str): Guidance for improvement.
            previous_feedback (str): Agent's previous feedback or instructions.
        Returns:
            str: Follow-up decision string indicating next action.
        """
        feedback = self.generative_model(
            prompt=f"{Constants.Instructions.AgenticAI.SystemModesTags.FOLLOW_UP_DECISION}\nYou were given this query: {query}\n.Based on your: reflection: {reflection}\nyour mistake: {mistake}\nyour own guidance: {guidance}\nyour previous feedback: {previous_feedback};\nDecide if further action is needed to answer the query."
        )
        return feedback

    def _update_memory_and_history(
        self,
        query,
        tool_decision,
        agent_feedback,
        response,
        reflection,
        mistake,
        guidance,
        errors,
    ):
        """
        Update the session memory and history file with the latest interaction.
        Args:
            query (str): User query.
            tool_decision (Dict[str, Any]): Tool decision dictionary.
            agent_feedback (str): Instructions for revising the response.
            response (str): Agent's generated response.
            reflection (str): Agent's self-reflection.
            mistake (str): Identified mistakes in the previous response.
            guidance (str): Guidance for improvement.
            errors (str): Any errors encountered during processing.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "tool_decision": tool_decision,
            "response": response,
            "reflection": reflection,
            "mistake": mistake,
            "guidance": guidance,
            "errors": errors,
            "agent_feedback": agent_feedback,
        }
        self.session_memory.append(entry)
        self._update_history_file("agent_history.json", entry)

    def _format_session_memory(self, memory) -> str:
        """Format session memory in an LLM-friendly way.
        Args:
            None
        Returns:
            str: Formatted session memory string.
        """
        if not memory:
            return ""

        formatted_parts = []
        formatted_parts.append("Previous attempts")
        for i, entry in enumerate(memory, 1):
            formatted_parts.append(f"\n--- Attempt {i} ---")
            formatted_parts.append(
                f"Tool used: {entry.get('tool_decision', {}).get('tool', 'none')}"
            )
            if entry.get("errors"):
                formatted_parts.append(f"⚠ Error: {entry['errors']}")
            formatted_parts.append(f"Response: {entry.get('response', '')[:100]}...")  # Truncated
            if entry.get("reflection"):
                formatted_parts.append(f"💡 Reflection: {entry['reflection']}")
            if entry.get("mistake"):
                formatted_parts.append(f"⚠ Mistake: {entry['mistake']}")
            if entry.get("guidance"):
                formatted_parts.append(f"→ Guidance: {entry['guidance']}")
            if entry.get("agent_feedback"):
                formatted_parts.append(f"🔄 Revision needed: {entry['agent_feedback']}")
        return "\n".join(formatted_parts)

    def agent_chat(self, query, max_tries=5):
        """Engage in a chat session with the agent, allowing for multiple attempts
        Args:
            query (str): User query.
            max_tries (int): Maximum number of attempts for the agent to refine its response.

        Returns:
            Tuple[str, str]: Final response and context used.
        """
        agent_feedback = ""
        self.query_memory = []
        for i in range(max_tries):
            response, tool_decision, context, errors = self._query_response(query, agent_feedback)
            reflection, mistake, guidance = self._reflect(
                query,
                tool_decision,
                response=response,
                errors=errors,
                agent_feedback=agent_feedback,
            )
            agent_feedback = self._agent_feedback(
                query,
                reflection,
                mistake,
                guidance,
                agent_feedback,
            )
            if Constants.Instructions.FeedbackTags.AFFIRM in agent_feedback:
                followup_decision = Constants.Instructions.FeedbackTags.AFFIRM
                agent_feedback = ""
            elif Constants.Instructions.FeedbackTags.FAILED in agent_feedback:
                followup_decision = Constants.Instructions.FeedbackTags.FAILED
                agent_feedback = ""
            elif Constants.Instructions.FeedbackTags.REVISE in agent_feedback:
                followup_decision = Constants.Instructions.FeedbackTags.REVISE
            else:
                # treating unexpected feedback format as affirm
                followup_decision = Constants.Instructions.FeedbackTags.AFFIRM
            self._update_memory_and_history(
                query,
                tool_decision,
                agent_feedback,
                response,
                reflection,
                mistake,
                guidance,
                errors,
            )
            self.query_memory.append(self.session_memory[-1])
            if followup_decision == Constants.Instructions.FeedbackTags.AFFIRM:
                break
            elif followup_decision == Constants.Instructions.FeedbackTags.FAILED:
                response = "✗ Agent decided that it cannot answer the query sufficiently."
                break
            elif followup_decision == Constants.Instructions.FeedbackTags.REVISE:
                continue
        return response, context
