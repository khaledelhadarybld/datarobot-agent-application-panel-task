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
from typing import Any, Optional, Union

from datarobot_genai.core.agents import (
    make_system_prompt,
)
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.config import Config
from agent.tools import analyze_data, calculate, generate_chart, remove_pii


class MyAgent(LangGraphAgent):
    """MyAgent is a custom data-processing agent with tools for PII removal,
    chart generation, data analysis, and mathematical calculations.

    It uses DataRobot's LLM Gateway or a specific deployment for language
    model interactions and exposes a single-node workflow that has access
    to all four tools.
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
                Defaults to None, in which case it will use the DATAROBOT_API_TOKEN environment variable.
            api_base: Optional[str]: Base URL for the DataRobot API.
                Defaults to None, in which case it will use the DATAROBOT_ENDPOINT environment variable.
            model: Optional[str]: The LLM model to use.
                Defaults to None.
            verbose: Optional[Union[bool, str]]: Whether to enable verbose logging.
                Accepts boolean or string values ("true"/"false"). Defaults to True.
            timeout: Optional[int]: How long to wait for the agent to respond.
                Defaults to 90 seconds.
            llm: Optional[BaseChatModel]: Pre-configured LLM instance provided by NAT.
                When set, llm() returns this directly instead of creating a ChatLiteLLM.
            workflow_tools: Optional[list[BaseTool]]: Additional tools from the workflow config (e.g. A2A client tools). Keyword-only.
            **kwargs: Any: Additional keyword arguments passed to the agent.

        Returns:
            None
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

    @property
    def tools(self) -> list[BaseTool]:
        """Return the list of tools available to the agent."""
        return [
            remove_pii,
            generate_chart,
            analyze_data,
            calculate,
        ]

    @property
    def workflow(self) -> StateGraph[MessagesState]:
        langgraph_workflow = StateGraph[
            MessagesState, None, MessagesState, MessagesState
        ](MessagesState)
        langgraph_workflow.add_node("agent_node", self.agent_node)
        langgraph_workflow.add_edge(START, "agent_node")
        langgraph_workflow.add_edge("agent_node", END)
        return langgraph_workflow

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful data-processing assistant with access to "
                    "specialized tools. Chat history is provided via {chat_history} "
                    "(it may be empty). Use it to stay consistent across turns.",
                ),
                (
                    "user",
                    "{user_prompt_content}",
                ),
            ]
        )

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
