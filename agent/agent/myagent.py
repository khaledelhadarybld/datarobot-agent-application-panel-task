# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-agent data-processing workflow.

Supervisor-based LangGraph workflow with:
  - intake_node         : classifies the user request into task categories
  - supervisor_node     : rule-based routing hub deciding which specialist runs next
  - analysis_node       : data analysis via pandas (analyze_data tool)
  - visualization_node  : chart generation (generate_chart tool)
  - calculation_node    : math evaluation (calculate tool)
  - pii_node            : PII detection & redaction (remove_pii tool)
  - presenter_node      : synthesises all results into a polished response
"""

from typing import Any, Literal, Optional, Union

from ag_ui.core import RunAgentInput
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.core.agents.base import extract_user_prompt_content
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.config import Config
from agent.tools import analyze_data, calculate, generate_chart, remove_pii

# ---------------------------------------------------------------------------
# Shared typed state
# ---------------------------------------------------------------------------


class DataProcessingState(MessagesState):
    """Shared state passed between all nodes in the data-processing workflow."""

    # Task classification flags
    needs_analysis: bool
    needs_visualization: bool
    needs_calculation: bool
    needs_pii_removal: bool

    # Sub-agent result buckets
    analysis_results: dict[str, Any]
    visualization_results: dict[str, Any]
    calculation_results: dict[str, Any]
    pii_results: dict[str, Any]

    # Workflow control
    next_agent: str  # "analysis" | "visualization" | "calculation" | "pii" | "FINISH"
    completed_steps: list[str]


# ---------------------------------------------------------------------------
# Pydantic model for structured intake extraction
# ---------------------------------------------------------------------------


class TaskClassification(BaseModel):
    """Structured classification of the user's data-processing request."""

    needs_analysis: bool = Field(
        description=(
            "True if the user wants to explore, summarise, filter, group, or "
            "statistically analyse tabular/JSON data."
        ),
    )
    needs_visualization: bool = Field(
        description=(
            "True if the user wants a chart, graph, plot, or any visual "
            "representation of data."
        ),
    )
    needs_calculation: bool = Field(
        description=(
            "True if the user wants to evaluate a mathematical expression, "
            "perform arithmetic, trigonometry, statistics, or linear algebra."
        ),
    )
    needs_pii_removal: bool = Field(
        description=(
            "True if the user wants to detect, redact, or remove personally "
            "identifiable information (emails, phones, SSNs, etc.) from text."
        ),
    )


# ---------------------------------------------------------------------------
# Conditional edge routing functions
# ---------------------------------------------------------------------------


def _route_supervisor(
    state: DataProcessingState,
) -> Literal[
    "analysis_node",
    "visualization_node",
    "calculation_node",
    "pii_node",
    "presenter_node",
]:
    next_agent = state.get("next_agent", "FINISH")
    mapping: dict[
        str,
        Literal[
            "analysis_node",
            "visualization_node",
            "calculation_node",
            "pii_node",
            "presenter_node",
        ],
    ] = {
        "analysis": "analysis_node",
        "visualization": "visualization_node",
        "calculation": "calculation_node",
        "pii": "pii_node",
        "FINISH": "presenter_node",
    }
    return mapping.get(next_agent, "presenter_node")


# ---------------------------------------------------------------------------
# MyAgent
# ---------------------------------------------------------------------------


class MyAgent(LangGraphAgent):
    """Multi-agent data-processing assistant.

    Orchestrates four specialist sub-agents (Analysis, Visualization,
    Calculation, PII) via a Supervisor node that uses deterministic
    conditional edges.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        verbose: Optional[Union[bool, str]] = True,
        timeout: Optional[int] = 90,
        *,
        llm: Optional[BaseChatModel] = None,
        workflow_tools: Optional[list[BaseTool]] = None,
        **kwargs: Any,
    ):
        """Initializes the MyAgent class.

        Args:
            api_key: Optional[str]: API key for authentication with DataRobot services.
            api_base: Optional[str]: Base URL for the DataRobot API.
            model: Optional[str]: The LLM model to use.
            verbose: Optional[Union[bool, str]]: Whether to enable verbose logging.
            timeout: Optional[int]: How long to wait for the agent to respond.
            llm: Optional[BaseChatModel]: Pre-configured LLM instance provided by NAT.
            workflow_tools: Optional[list[BaseTool]]: Additional tools from the workflow config.
            **kwargs: Any: Additional keyword arguments passed to the agent.
        """
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )
        self._nat_llm = llm
        self._workflow_tools = workflow_tools or []
        self.config = Config()
        self.default_model = self.config.llm_default_model
        if model in ("unknown", "datarobot-deployed-llm"):
            self.model = self.default_model

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[BaseTool]:
        """Return the list of all tools available to the agent."""
        return [
            remove_pii,
            generate_chart,
            analyze_data,
            calculate,
        ]

    # ------------------------------------------------------------------
    # LLM factory — MUST NOT be modified (DataRobot requirement)
    # ------------------------------------------------------------------

    def llm(
        self,
        auto_model_override: bool = True,
    ) -> BaseChatModel:
        """Returns the LLM to use for agent nodes.

        In NAT mode, returns the pre-configured LLM provided at construction.
        In DRUM mode, creates a ChatLiteLLM using the configured API credentials.

        Args:
            auto_model_override: Optional[bool]: If True, it will try and use the model
                specified in the request but automatically back out if the LLM Gateway is
                not available.

        Returns:
            BaseChatModel: The model to use.
        """
        if self._nat_llm is not None:
            return self._nat_llm

        api_base = self.litellm_api_base(self.config.llm_deployment_id)
        model = self.model or self.default_model
        if auto_model_override and not self.config.use_datarobot_llm_gateway:
            model = self.default_model
        if self.verbose:
            print(f"Using model: {model}")

        config = {
            "model": model,
            "api_base": api_base,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "streaming": True,
            "max_retries": 3,
        }

        if not self.config.use_datarobot_llm_gateway and self._identity_header:
            config["model_kwargs"] = {"extra_headers": self._identity_header}  # type: ignore[assignment]

        return ChatLiteLLM(**config)

    # ------------------------------------------------------------------
    # Prompt template
    # ------------------------------------------------------------------

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful data-processing assistant with access to "
                    "specialized tools for data analysis, chart generation, math "
                    "calculations, and PII removal.",
                ),
                ("user", "{user_prompt_content}"),
            ]
        )

    # ------------------------------------------------------------------
    # Convert AG-UI input messages into LangGraph state
    # ------------------------------------------------------------------

    def convert_input_message(self, run_agent_input: RunAgentInput) -> Command[Any]:
        """Convert the full AG-UI message history into LangGraph state messages.

        Converts every prior turn in run_agent_input.messages into the
        appropriate HumanMessage / AIMessage so that _intake_node sees the
        complete conversation history, not just the latest user turn.
        """
        history_messages: list[Any] = []
        all_messages = list(run_agent_input.messages or [])

        # Prior messages become raw history entries.
        # The last user message is formatted through the prompt template so
        # the system prompt is included.
        prior_messages = all_messages[:-1] if len(all_messages) > 1 else []

        for msg in prior_messages:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "") or ""
            if role == "user":
                history_messages.append(HumanMessage(content=str(content)))
            elif role == "assistant":
                history_messages.append(AIMessage(content=str(content)))

        raw = extract_user_prompt_content(run_agent_input)
        user_prompt_str = raw if isinstance(raw, str) else str(raw)
        current_messages = self.prompt_template.invoke(
            {"user_prompt_content": user_prompt_str}
        ).to_messages()

        return Command(update={"messages": history_messages + current_messages})

    # ------------------------------------------------------------------
    # Node: intake_node
    # Classifies the user request into task categories.
    # ------------------------------------------------------------------

    def _intake_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Classify the user's request to determine which specialist agents are needed."""
        # Use keyword-based classification to avoid LLM streaming in this node.
        # LLM calls in non-terminal nodes create orphaned TEXT_MESSAGE_START events
        # that the AG-UI protocol cannot close before RUN_FINISHED.
        user_message = _extract_latest_user_message(state).lower()

        needs_analysis = any(
            kw in user_message
            for kw in [
                "analyze",
                "analyse",
                "data",
                "csv",
                "json",
                "dataframe",
                "describe",
                "filter",
                "groupby",
                "group by",
                "statistics",
                "head",
                "tail",
                "correlation",
                "sort",
                "column",
                "row",
                "value_counts",
                "nunique",
                "sample",
                "shape",
                "info",
                "mean",
                "median",
                "sum of",
                "min of",
                "max of",
            ]
        )
        needs_visualization = any(
            kw in user_message
            for kw in [
                "chart",
                "graph",
                "plot",
                "visualize",
                "visualise",
                "bar chart",
                "line chart",
                "pie chart",
                "scatter",
                "histogram",
                "draw",
            ]
        )
        needs_calculation = any(
            kw in user_message
            for kw in [
                "calculate",
                "compute",
                "math",
                "equation",
                "formula",
                "sqrt",
                "sin",
                "cos",
                "tan",
                "log",
                "log2",
                "log10",
                "multiply",
                "divide",
                "add",
                "subtract",
                "factorial",
                "np_mean",
                "np_median",
                "np_std",
                "np_dot",
            ]
        )
        needs_pii = any(
            kw in user_message
            for kw in [
                "pii",
                "redact",
                "remove pii",
                "personal information",
                "ssn",
                "social security",
                "credit card",
                "anonymize",
                "anonymise",
                "phone number",
                "email address",
            ]
        )

        classification = TaskClassification(
            needs_analysis=needs_analysis,
            needs_visualization=needs_visualization,
            needs_calculation=needs_calculation,
            needs_pii_removal=needs_pii,
        )

        if self.verbose:
            print(f"[intake_node] classification: {classification.model_dump()}")

        return {
            "needs_analysis": classification.needs_analysis,
            "needs_visualization": classification.needs_visualization,
            "needs_calculation": classification.needs_calculation,
            "needs_pii_removal": classification.needs_pii_removal,
            "analysis_results": state.get("analysis_results", {}),
            "visualization_results": state.get("visualization_results", {}),
            "calculation_results": state.get("calculation_results", {}),
            "pii_results": state.get("pii_results", {}),
            "completed_steps": state.get("completed_steps", []),
            "next_agent": state.get("next_agent", ""),
        }

    # ------------------------------------------------------------------
    # Node: supervisor_node
    # Deterministic routing based on classification flags and completed steps.
    # ------------------------------------------------------------------

    def _supervisor_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Route to the next specialist agent based on what's needed and what's done."""
        completed = state.get("completed_steps", [])

        needs_analysis = state.get("needs_analysis", False)
        needs_visualization = state.get("needs_visualization", False)
        needs_calculation = state.get("needs_calculation", False)
        needs_pii = state.get("needs_pii_removal", False)

        # Deterministic priority order: analysis → visualization → calculation → pii
        if needs_analysis and "analysis" not in completed:
            next_agent = "analysis"
            reasoning = "Data analysis is needed and has not been completed yet."
        elif needs_visualization and "visualization" not in completed:
            next_agent = "visualization"
            reasoning = "Visualization is needed and has not been completed yet."
        elif needs_calculation and "calculation" not in completed:
            next_agent = "calculation"
            reasoning = "Math calculation is needed and has not been completed yet."
        elif needs_pii and "pii" not in completed:
            next_agent = "pii"
            reasoning = "PII removal is needed and has not been completed yet."
        else:
            next_agent = "FINISH"
            reasoning = "All required tasks have been completed."

        if self.verbose:
            print(f"[supervisor] → {next_agent}: {reasoning}")

        return {
            "next_agent": next_agent,
            "completed_steps": completed,
        }

    # ------------------------------------------------------------------
    # Specialist sub-agent factories
    # ------------------------------------------------------------------

    @property
    def _analysis_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[analyze_data] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Data Analysis Agent. Your job is to explore and "
                "analyse data using the analyze_data tool.\n\n"
                "Capabilities:\n"
                "- Load JSON data into a pandas DataFrame\n"
                "- Perform operations: describe, head, tail, info, shape, columns, "
                "dtypes, value_counts, correlation, sort, filter, groupby, mean, "
                "median, sum, min, max, nunique, sample\n\n"
                "Always call the analyze_data tool with the appropriate operation. "
                "Return a clear summary of your findings."
            ),
            name="analysis_agent",
        )

    @property
    def _visualization_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[generate_chart] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Visualization Agent. Your job is to create charts "
                "and graphs from data using the generate_chart tool.\n\n"
                "Supported chart types: bar, line, pie, scatter, histogram.\n\n"
                "Data format:\n"
                '- For bar/line/scatter: {"x": [...], "y": [...]}\n'
                '- For pie: {"labels": [...], "values": [...]}\n'
                '- For histogram: {"values": [...]}\n\n'
                "Always call the generate_chart tool. Include a title and axis "
                "labels when appropriate. Return the chart image."
            ),
            name="visualization_agent",
        )

    @property
    def _calculation_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[calculate] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Math Calculation Agent. Your job is to evaluate "
                "mathematical expressions using the calculate tool.\n\n"
                "Capabilities:\n"
                "- Arithmetic: +, -, *, /, **, %\n"
                "- Trigonometry: sin, cos, tan, asin, acos, atan\n"
                "- Logarithms: log, log2, log10\n"
                "- Constants: pi, e, inf\n"
                "- Statistics (numpy): np_mean, np_median, np_std, np_var\n"
                "- Linear algebra: np_dot, np_cross, np_linalg_norm, "
                "np_linalg_det, np_linalg_inv\n\n"
                "Always call the calculate tool. Explain the result clearly."
            ),
            name="calculation_agent",
        )

    @property
    def _pii_agent(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[remove_pii] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the PII Removal Agent. Your job is to detect and redact "
                "personally identifiable information using the remove_pii tool.\n\n"
                "Detectable PII types:\n"
                "- Email addresses\n"
                "- Phone numbers\n"
                "- Social Security Numbers (SSNs)\n"
                "- Credit card numbers\n"
                "- IP addresses\n"
                "- Dates of birth\n\n"
                "Always call the remove_pii tool with the text to scan. "
                "Return the redacted text and a summary of what was found."
            ),
            name="pii_agent",
        )

    # ------------------------------------------------------------------
    # Sub-agent node wrappers
    # ------------------------------------------------------------------

    def _analysis_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Run the analysis sub-agent and persist results into shared state."""
        user_request = _extract_latest_user_message(state)
        prior_results = _gather_prior_results(state)

        context_msg = HumanMessage(
            content=(
                f"The user asked:\n{user_request}\n\n"
                f"{prior_results}"
                "Please analyse the data as requested using the analyze_data tool."
            )
        )
        result = self._analysis_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)

        completed = list(state.get("completed_steps", []))
        if "analysis" not in completed:
            completed.append("analysis")

        return {
            "analysis_results": {"summary": last_ai},
            "completed_steps": completed,
        }

    def _visualization_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Run the visualization sub-agent and persist results into shared state."""
        user_request = _extract_latest_user_message(state)
        prior_results = _gather_prior_results(state)

        context_msg = HumanMessage(
            content=(
                f"The user asked:\n{user_request}\n\n"
                f"{prior_results}"
                "Please create the requested chart using the generate_chart tool."
            )
        )
        result = self._visualization_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)

        completed = list(state.get("completed_steps", []))
        if "visualization" not in completed:
            completed.append("visualization")

        return {
            "visualization_results": {"summary": last_ai},
            "completed_steps": completed,
        }

    def _calculation_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Run the calculation sub-agent and persist results into shared state."""
        user_request = _extract_latest_user_message(state)
        prior_results = _gather_prior_results(state)

        context_msg = HumanMessage(
            content=(
                f"The user asked:\n{user_request}\n\n"
                f"{prior_results}"
                "Please evaluate the mathematical expression using the calculate tool."
            )
        )
        result = self._calculation_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)

        completed = list(state.get("completed_steps", []))
        if "calculation" not in completed:
            completed.append("calculation")

        return {
            "calculation_results": {"summary": last_ai},
            "completed_steps": completed,
        }

    def _pii_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Run the PII removal sub-agent and persist results into shared state."""
        user_request = _extract_latest_user_message(state)

        context_msg = HumanMessage(
            content=(
                f"The user asked:\n{user_request}\n\n"
                "Please scan the text for PII and redact it using the remove_pii tool."
            )
        )
        result = self._pii_agent.invoke({"messages": [context_msg]})
        last_ai = _last_ai_content(result)

        completed = list(state.get("completed_steps", []))
        if "pii" not in completed:
            completed.append("pii")

        return {
            "pii_results": {"summary": last_ai},
            "completed_steps": completed,
        }

    # ------------------------------------------------------------------
    # Node: presenter_node
    # Synthesises all sub-agent results into a single user-facing response.
    # ------------------------------------------------------------------

    def _presenter_node(self, state: DataProcessingState) -> dict[str, Any]:
        """Combine all specialist results into a polished, user-facing response.

        When no specialists were needed (e.g. a simple greeting), responds
        conversationally to the user's message instead of trying to synthesise
        empty results.
        """
        analysis = state.get("analysis_results", {}).get("summary", "")
        visualization = state.get("visualization_results", {}).get("summary", "")
        calculation = state.get("calculation_results", {}).get("summary", "")
        pii = state.get("pii_results", {}).get("summary", "")

        sections: list[str] = []
        if analysis:
            sections.append(f"Data Analysis Results:\n{analysis}")
        if visualization:
            sections.append(f"Visualization Results:\n{visualization}")
        if calculation:
            sections.append(f"Calculation Results:\n{calculation}")
        if pii:
            sections.append(f"PII Removal Results:\n{pii}")

        # If no specialists produced results, respond conversationally
        if not sections:
            user_message = _extract_latest_user_message(state)
            conversational_prompt = (
                "You are a friendly, helpful data-processing assistant. "
                "The user sent a conversational message that does not require "
                "any data analysis, charting, math, or PII removal.\n\n"
                "Respond naturally and warmly. Briefly mention what you can help "
                "with (data analysis, chart generation, math calculations, and "
                "PII removal) so the user knows your capabilities.\n\n"
                "Keep your response short and friendly — 2-3 sentences max.\n\n"
                f"User message: {user_message}"
            )
            self.llm().invoke(conversational_prompt)
        else:
            combined = "\n\n".join(sections)
            synthesis_prompt = (
                "You are a helpful data-processing assistant presenting results to the user.\n\n"
                "Below are the results from specialist agents that processed the user's request. "
                "Synthesise them into a single, clear, well-formatted markdown response.\n\n"
                "Rules:\n"
                "- Do NOT expose internal agent names or technical routing details.\n"
                "- Present results naturally as if you did all the work yourself.\n"
                "- If a chart image (base64) is included, preserve the markdown image tag exactly.\n"
                "- Explain results clearly and concisely.\n"
                "- Use headers and formatting for readability.\n\n"
                f"Specialist results:\n{combined}"
            )
            self.llm().invoke(synthesis_prompt)

        return {}

    # ------------------------------------------------------------------
    # Workflow graph
    # ------------------------------------------------------------------

    @property
    def workflow(self) -> StateGraph[DataProcessingState]:  # type: ignore[override]
        graph: StateGraph[DataProcessingState] = StateGraph(DataProcessingState)

        # Register nodes
        graph.add_node("intake_node", self._intake_node)
        graph.add_node("supervisor_node", self._supervisor_node)
        graph.add_node("analysis_node", self._analysis_node)
        graph.add_node("visualization_node", self._visualization_node)
        graph.add_node("calculation_node", self._calculation_node)
        graph.add_node("pii_node", self._pii_node)
        graph.add_node("presenter_node", self._presenter_node)

        # Entry point
        graph.add_edge(START, "intake_node")

        # Intake always goes to supervisor
        graph.add_edge("intake_node", "supervisor_node")

        # Supervisor → specialist or presenter (conditional)
        graph.add_conditional_edges(
            "supervisor_node",
            _route_supervisor,
            {
                "analysis_node": "analysis_node",
                "visualization_node": "visualization_node",
                "calculation_node": "calculation_node",
                "pii_node": "pii_node",
                "presenter_node": "presenter_node",
            },
        )

        # All specialist nodes return to supervisor after completion
        graph.add_edge("analysis_node", "supervisor_node")
        graph.add_edge("visualization_node", "supervisor_node")
        graph.add_edge("calculation_node", "supervisor_node")
        graph.add_edge("pii_node", "supervisor_node")

        # Presenter is the final node
        graph.add_edge("presenter_node", END)

        return graph

    # ------------------------------------------------------------------
    # Keep agent_node for backward compatibility (single-agent fallback)
    # ------------------------------------------------------------------

    @property
    def agent_node(self) -> Any:
        return create_agent(
            self.llm(),
            tools=self.tools + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are a powerful data-processing assistant with four specialized tools.\n"
                "\n"
                "## Your Tools\n"
                "\n"
                "1. **PII Remover** (`remove_pii`): Detects and redacts personally "
                "identifiable information from text, including emails, phone numbers, "
                "SSNs, credit card numbers, IP addresses, and dates of birth.\n"
                "\n"
                "2. **Chart Generator** (`generate_chart`): Creates charts (bar, line, "
                "pie, scatter, histogram) from JSON data and returns them as base64 "
                "PNG images.\n"
                "\n"
                "3. **Data Analyzer** (`analyze_data`): Loads JSON data into a pandas "
                "DataFrame and performs analysis operations like describe, head, filter, "
                "groupby, value_counts, correlation, sort, and more.\n"
                "\n"
                "4. **Math Calculator** (`calculate`): Evaluates mathematical expressions "
                "safely using Python's math library and numpy. Supports arithmetic, "
                "trigonometry, logarithms, statistics, and linear algebra.\n"
                "\n"
                "## Guidelines\n"
                "\n"
                "- Use the appropriate tool for each task.\n"
                "- When the user provides data, use the data analyzer or chart generator.\n"
                "- When the user asks for calculations, use the math calculator.\n"
                "- When the user asks to clean or redact PII, use the PII remover.\n"
                "- You can chain tools: e.g., analyze data, then chart the results.\n"
                "- Always explain what you did and present results clearly.\n"
            ),
            name="agent",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_ai_content(result: Any) -> str:
    """Extract the last AI message content from a react-agent result dict."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return str(result)


def _extract_latest_user_message(state: DataProcessingState) -> str:
    """Extract the latest user message from the conversation."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _gather_prior_results(state: DataProcessingState) -> str:
    """Collect results from previously completed specialist agents."""
    parts: list[str] = []
    analysis = state.get("analysis_results", {}).get("summary", "")
    if analysis:
        parts.append(f"Previous analysis results:\n{analysis}")
    visualization = state.get("visualization_results", {}).get("summary", "")
    if visualization:
        parts.append(f"Previous visualization results:\n{visualization}")
    calculation = state.get("calculation_results", {}).get("summary", "")
    if calculation:
        parts.append(f"Previous calculation results:\n{calculation}")
    pii = state.get("pii_results", {}).get("summary", "")
    if pii:
        parts.append(f"Previous PII removal results:\n{pii}")
    if parts:
        return "\n\n".join(parts) + "\n\n"
    return ""
