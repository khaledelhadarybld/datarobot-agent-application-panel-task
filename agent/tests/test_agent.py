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

import os
from unittest.mock import ANY, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from agent import MyAgent
from agent.myagent import (
    DataProcessingState,
    TaskClassification,
    _extract_latest_user_message,
    _gather_prior_results,
    _last_ai_content,
    _route_supervisor,
)


class TestMyAgentLangGraph:
    @pytest.fixture
    def agent(self) -> MyAgent:
        return MyAgent(
            api_key="test_key",
            api_base="test_base",
            verbose=True,
            model="datarobot/azure/gpt-5-mini-2025-08-07",
        )

    def test_init_with_explicit_parameters(self):
        """Test initialization with explicitly provided parameters."""
        api_key = "test-api-key"
        api_base = "https://test-api-base.com"
        model = "test-model"
        verbose = True

        agent = MyAgent(
            api_key=api_key, api_base=api_base, model=model, verbose=verbose
        )

        assert agent.api_key == api_key
        assert agent.api_base == api_base
        assert agent.model == model
        assert agent.verbose is True

    @patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "env-api-key",
            "DATAROBOT_ENDPOINT": "https://env-api-base.com",
        },
    )
    def test_init_with_environment_variables(self):
        """Test initialization using environment variables when no explicit parameters."""
        agent = MyAgent()

        assert agent.api_key == "env-api-key"
        assert agent.api_base == "https://env-api-base.com"
        assert agent.model is None
        assert agent.verbose is True

    @patch.dict(
        os.environ,
        {
            "DATAROBOT_API_TOKEN": "env-api-key",
            "DATAROBOT_ENDPOINT": "https://env-api-base.com",
        },
    )
    def test_init_explicit_params_override_env_vars(self):
        """Test explicit parameters override environment variables."""
        api_key = "explicit-api-key"
        api_base = "https://explicit-api-base.com"

        agent = MyAgent(api_key=api_key, api_base=api_base)

        assert agent.api_key == "explicit-api-key"
        assert agent.api_base == "https://explicit-api-base.com"

    def test_init_with_string_verbose_true(self):
        """Test initialization with string 'true' for verbose parameter."""
        verbose_values = ["true", "TRUE", "True"]

        for verbose in verbose_values:
            agent = MyAgent(verbose=verbose)
            assert agent.verbose is True

    def test_init_with_string_verbose_false(self):
        """Test initialization with string 'false' for verbose parameter."""
        verbose_values = ["false", "FALSE", "False"]

        for verbose in verbose_values:
            agent = MyAgent(verbose=verbose)
            assert agent.verbose is False

    def test_init_with_boolean_verbose(self):
        """Test initialization with boolean values for verbose parameter."""
        agent = MyAgent(verbose=True)
        assert agent.verbose is True

        agent = MyAgent(verbose=False)
        assert agent.verbose is False

    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_additional_kwargs(self):
        """Test initialization with additional keyword arguments."""
        additional_kwargs = {"extra_param1": "value1", "extra_param2": 42}

        agent = MyAgent(**additional_kwargs)

        assert agent.api_key is None
        assert agent.api_base == "https://app.datarobot.com"
        assert agent.model is None
        assert agent.verbose is True

        with pytest.raises(AttributeError):
            _ = agent.extra_param1

    @pytest.mark.parametrize(
        "api_base,expected_result",
        [
            ("https://example.com", "https://example.com/"),
            ("https://example.com/", "https://example.com/"),
            ("https://example.com/api/v2", "https://example.com/"),
            ("https://example.com/api/v2/", "https://example.com/"),
            ("https://example.com/other-path", "https://example.com/other-path/"),
            (
                "https://custom.example.com:8080/path/to/api/v2/",
                "https://custom.example.com:8080/path/to/",
            ),
            (
                "https://example.com/api/v2/deployment/",
                "https://example.com/api/v2/deployment/",
            ),
            (
                "https://example.com/api/v2/deployment",
                "https://example.com/api/v2/deployment/",
            ),
            (
                "https://example.com/api/v2/genai/llmgw/chat/completions",
                "https://example.com/api/v2/genai/llmgw/chat/completions",
            ),
            (
                "https://example.com/api/v2/genai/llmgw/chat/completions/",
                "https://example.com/api/v2/genai/llmgw/chat/completions",
            ),
            (None, "https://app.datarobot.com/"),
        ],
    )
    @patch("agent.myagent.ChatLiteLLM")
    def test_llm_gateway_with_api_base(self, mock_llm, api_base, expected_result):
        """Test api_base_litellm property with various URL formats."""
        with patch.dict(os.environ, {}, clear=True):
            agent = MyAgent(
                api_base=api_base, model="datarobot/azure/gpt-5-mini-2025-08-07"
            )
            agent.config.use_datarobot_llm_gateway = True
            _ = agent.llm()
            mock_llm.assert_called_once_with(
                model="datarobot/azure/gpt-5-mini-2025-08-07",
                api_base=expected_result,
                api_key=None,
                timeout=90,
                streaming=True,
                max_retries=3,
            )

    @pytest.mark.parametrize(
        "api_base,expected_result",
        [
            (
                "https://example.com",
                "https://example.com/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://example.com/",
                "https://example.com/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://example.com/api/v2/",
                "https://example.com/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://example.com/api/v2",
                "https://example.com/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://example.com/other-path",
                "https://example.com/other-path/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://custom.example.com:8080/path/to",
                "https://custom.example.com:8080/path/to/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://custom.example.com:8080/path/to/api/v2/",
                "https://custom.example.com:8080/path/to/api/v2/deployments/test-id/chat/completions",
            ),
            (
                "https://example.com/api/v2/deployments/",
                "https://example.com/api/v2/deployments/",
            ),
            (
                "https://example.com/api/v2/deployments",
                "https://example.com/api/v2/deployments/",
            ),
            (
                "https://example.com/api/v2/genai/llmgw/chat/completions",
                "https://example.com/api/v2/genai/llmgw/chat/completions",
            ),
            (
                "https://example.com/api/v2/genai/llmgw/chat/completions/",
                "https://example.com/api/v2/genai/llmgw/chat/completions",
            ),
            (
                None,
                "https://app.datarobot.com/api/v2/deployments/test-id/chat/completions",
            ),
        ],
    )
    @patch("agent.myagent.ChatLiteLLM")
    def test_llm_deployment_with_api_base(self, mock_llm, api_base, expected_result):
        """Test api_base_litellm property with various URL formats."""
        with patch.dict(os.environ, {"LLM_DEPLOYMENT_ID": "test-id"}, clear=True):
            agent = MyAgent(api_base=api_base)
            agent.config.llm_default_model = "datarobot/azure/gpt-5-mini-2025-08-07"
            _ = agent.llm()
            mock_llm.assert_called_once_with(
                model="datarobot/azure/gpt-5-mini-2025-08-07",
                api_base=expected_result,
                api_key=None,
                timeout=90,
                streaming=True,
                max_retries=3,
            )

    @patch("agent.myagent.ChatLiteLLM")
    def test_llm(self, mock_llm, agent):
        # Test that ChatLiteLLM is created with correct parameters
        agent.llm()
        mock_llm.assert_called_once_with(
            model="datarobot/azure/gpt-5-mini-2025-08-07",
            api_base="test_base/",
            api_key="test_key",
            timeout=90,
            streaming=True,
            max_retries=3,
        )

    @patch("agent.myagent.ChatLiteLLM")
    def test_llm_property_with_no_api_base(self, mock_llm, agent):
        # Test that ChatLiteLLM is created with correct parameters
        with patch.dict(os.environ, {}, clear=True):
            agent = MyAgent(
                api_key="test_key",
                verbose=True,
                model="datarobot/azure/gpt-5-mini-2025-08-07",
            )
            agent.llm()
            mock_llm.assert_called_once_with(
                model="datarobot/azure/gpt-5-mini-2025-08-07",
                api_base="https://app.datarobot.com/",
                api_key="test_key",
                timeout=90,
                streaming=True,
                max_retries=3,
            )

    @patch("agent.myagent.ChatLiteLLM")
    @pytest.mark.parametrize("use_datarobot_llm_gateway", [True, False])
    def test_llm_with_identity_token(self, mock_llm, use_datarobot_llm_gateway):
        with patch.dict(os.environ, {"LLM_DEPLOYMENT_ID": "test-id"}, clear=True):
            agent = MyAgent(
                api_key="test_key",
                verbose=True,
                model="datarobot/azure/gpt-5-mini-2025-08-07",
                forwarded_headers={
                    "x-datarobot-api-key": "abc",
                    "x-datarobot-identity-token": "xyz",
                },
            )
            agent.config.use_datarobot_llm_gateway = use_datarobot_llm_gateway
            agent.llm()

            if use_datarobot_llm_gateway:
                mock_llm.assert_called_once_with(
                    model="datarobot/azure/gpt-5-mini-2025-08-07",
                    api_base="https://app.datarobot.com/api/v2/deployments/test-id/chat/completions",
                    api_key="test_key",
                    timeout=90,
                    streaming=True,
                    max_retries=3,
                )
            else:
                mock_llm.assert_called_once_with(
                    model="datarobot/azure/gpt-5-mini-2025-08-07",
                    api_base="https://app.datarobot.com/api/v2/deployments/test-id/chat/completions",
                    api_key="test_key",
                    timeout=90,
                    streaming=True,
                    max_retries=3,
                    model_kwargs={
                        "extra_headers": {"X-DataRobot-Identity-Token": "xyz"}
                    },
                )

    # ------------------------------------------------------------------
    # Agent node tests (backward-compatible single agent_node)
    # ------------------------------------------------------------------

    @patch("agent.myagent.create_agent")
    def test_agent_node_property(self, mock_create_agent, agent):
        """Test that agent_node creates a react agent with all tools."""
        mock_llm = Mock()
        with patch.object(MyAgent, "llm", return_value=mock_llm):
            _ = agent.agent_node
            mock_create_agent.assert_called_once_with(
                mock_llm,
                tools=ANY,
                system_prompt=ANY,
                name="agent",
            )

    @patch("agent.myagent.create_agent")
    def test_agent_node_includes_workflow_tools(self, mock_create_agent):
        """Test that agent_node includes workflow_tools alongside mcp_tools."""
        extra_tool = Mock()
        agent = MyAgent(
            api_key="test_key",
            api_base="test_base",
            workflow_tools=[extra_tool],
        )
        mock_llm = Mock()
        with patch.object(MyAgent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent.agent_node
                _, kwargs = mock_create_agent.call_args
                assert extra_tool in kwargs["tools"]

    # ------------------------------------------------------------------
    # Tool property tests
    # ------------------------------------------------------------------

    def test_tools_property(self, agent):
        """Test that tools returns all four tools."""
        tools = agent.tools
        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "remove_pii",
            "generate_chart",
            "analyze_data",
            "calculate",
        }

    # ------------------------------------------------------------------
    # Workflow and template tests
    # ------------------------------------------------------------------

    def test_workflow_property(self, agent):
        """Test that workflow returns a StateGraph with correct multi-agent structure."""
        mock_llm = Mock()
        with patch.object(MyAgent, "llm", return_value=mock_llm):
            with patch("agent.myagent.create_agent"):
                workflow = agent.workflow
                assert workflow is not None
                # Verify all multi-agent nodes are present
                expected_nodes = {
                    "intake_node",
                    "supervisor_node",
                    "analysis_node",
                    "visualization_node",
                    "calculation_node",
                    "pii_node",
                    "presenter_node",
                }
                assert expected_nodes == set(workflow.nodes.keys())

    def test_prompt_template_property(self, agent):
        """Test that prompt_template returns a ChatPromptTemplate."""
        template = agent.prompt_template
        assert template is not None
        assert isinstance(template, ChatPromptTemplate)

    # ------------------------------------------------------------------
    # Intake node tests
    # ------------------------------------------------------------------

    def test_intake_node_classifies_request(self, agent):
        """Test that intake_node calls structured LLM and returns classification."""
        mock_classification = TaskClassification(
            needs_analysis=True,
            needs_visualization=True,
            needs_calculation=False,
            needs_pii_removal=False,
        )
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_classification

        mock_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch.object(agent, "llm", return_value=mock_llm):
            state: DataProcessingState = {  # type: ignore[assignment]
                "messages": [HumanMessage(content="Analyse this data and chart it")],
                "needs_analysis": False,
                "needs_visualization": False,
                "needs_calculation": False,
                "needs_pii_removal": False,
                "analysis_results": {},
                "visualization_results": {},
                "calculation_results": {},
                "pii_results": {},
                "next_agent": "",
                "completed_steps": [],
            }
            result = agent._intake_node(state)

            assert result["needs_analysis"] is True
            assert result["needs_visualization"] is True
            assert result["needs_calculation"] is False
            assert result["needs_pii_removal"] is False

    # ------------------------------------------------------------------
    # Supervisor node tests
    # ------------------------------------------------------------------

    def test_supervisor_routes_to_analysis_first(self, agent):
        """Test supervisor routes to analysis when it's needed and not done."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "needs_analysis": True,
            "needs_visualization": True,
            "needs_calculation": False,
            "needs_pii_removal": False,
            "completed_steps": [],
        }
        result = agent._supervisor_node(state)
        assert result["next_agent"] == "analysis"

    def test_supervisor_routes_to_visualization_after_analysis(self, agent):
        """Test supervisor routes to visualization after analysis is done."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "needs_analysis": True,
            "needs_visualization": True,
            "needs_calculation": False,
            "needs_pii_removal": False,
            "completed_steps": ["analysis"],
        }
        result = agent._supervisor_node(state)
        assert result["next_agent"] == "visualization"

    def test_supervisor_routes_to_calculation(self, agent):
        """Test supervisor routes to calculation when needed."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "needs_analysis": False,
            "needs_visualization": False,
            "needs_calculation": True,
            "needs_pii_removal": False,
            "completed_steps": [],
        }
        result = agent._supervisor_node(state)
        assert result["next_agent"] == "calculation"

    def test_supervisor_routes_to_pii(self, agent):
        """Test supervisor routes to PII when needed."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "needs_analysis": False,
            "needs_visualization": False,
            "needs_calculation": False,
            "needs_pii_removal": True,
            "completed_steps": [],
        }
        result = agent._supervisor_node(state)
        assert result["next_agent"] == "pii"

    def test_supervisor_routes_to_finish_when_all_done(self, agent):
        """Test supervisor routes to FINISH when all needed tasks are completed."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "needs_analysis": True,
            "needs_visualization": True,
            "needs_calculation": False,
            "needs_pii_removal": False,
            "completed_steps": ["analysis", "visualization"],
        }
        result = agent._supervisor_node(state)
        assert result["next_agent"] == "FINISH"

    def test_supervisor_routes_to_finish_when_nothing_needed(self, agent):
        """Test supervisor routes to FINISH when no tasks are needed."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "needs_analysis": False,
            "needs_visualization": False,
            "needs_calculation": False,
            "needs_pii_removal": False,
            "completed_steps": [],
        }
        result = agent._supervisor_node(state)
        assert result["next_agent"] == "FINISH"

    # ------------------------------------------------------------------
    # Route supervisor function tests
    # ------------------------------------------------------------------

    def test_route_supervisor_analysis(self):
        """Test _route_supervisor returns analysis_node."""
        state: DataProcessingState = {"next_agent": "analysis"}  # type: ignore[assignment]
        assert _route_supervisor(state) == "analysis_node"

    def test_route_supervisor_visualization(self):
        """Test _route_supervisor returns visualization_node."""
        state: DataProcessingState = {"next_agent": "visualization"}  # type: ignore[assignment]
        assert _route_supervisor(state) == "visualization_node"

    def test_route_supervisor_calculation(self):
        """Test _route_supervisor returns calculation_node."""
        state: DataProcessingState = {"next_agent": "calculation"}  # type: ignore[assignment]
        assert _route_supervisor(state) == "calculation_node"

    def test_route_supervisor_pii(self):
        """Test _route_supervisor returns pii_node."""
        state: DataProcessingState = {"next_agent": "pii"}  # type: ignore[assignment]
        assert _route_supervisor(state) == "pii_node"

    def test_route_supervisor_finish(self):
        """Test _route_supervisor returns presenter_node for FINISH."""
        state: DataProcessingState = {"next_agent": "FINISH"}  # type: ignore[assignment]
        assert _route_supervisor(state) == "presenter_node"

    def test_route_supervisor_unknown_defaults_to_presenter(self):
        """Test _route_supervisor defaults to presenter_node for unknown values."""
        state: DataProcessingState = {"next_agent": "unknown"}  # type: ignore[assignment]
        assert _route_supervisor(state) == "presenter_node"

    # ------------------------------------------------------------------
    # Specialist sub-agent factory tests
    # ------------------------------------------------------------------

    @patch("agent.myagent.create_agent")
    def test_analysis_agent_created_with_correct_tool(self, mock_create_agent, agent):
        """Test that _analysis_agent is created with analyze_data tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._analysis_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "analyze_data" in tool_names
                assert kwargs["name"] == "analysis_agent"

    @patch("agent.myagent.create_agent")
    def test_visualization_agent_created_with_correct_tool(
        self, mock_create_agent, agent
    ):
        """Test that _visualization_agent is created with generate_chart tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._visualization_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "generate_chart" in tool_names
                assert kwargs["name"] == "visualization_agent"

    @patch("agent.myagent.create_agent")
    def test_calculation_agent_created_with_correct_tool(
        self, mock_create_agent, agent
    ):
        """Test that _calculation_agent is created with calculate tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._calculation_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "calculate" in tool_names
                assert kwargs["name"] == "calculation_agent"

    @patch("agent.myagent.create_agent")
    def test_pii_agent_created_with_correct_tool(self, mock_create_agent, agent):
        """Test that _pii_agent is created with remove_pii tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._pii_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "remove_pii" in tool_names
                assert kwargs["name"] == "pii_agent"

    # ------------------------------------------------------------------
    # Sub-agent node wrapper tests
    # ------------------------------------------------------------------

    def test_analysis_node_stores_results(self, agent):
        """Test that _analysis_node invokes the agent and stores results."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Analysis complete: 5 rows, 3 columns")]
        }

        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="Analyse this data")],
            "completed_steps": [],
            "analysis_results": {},
            "visualization_results": {},
            "calculation_results": {},
            "pii_results": {},
        }

        with patch.object(
            type(agent),
            "_analysis_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._analysis_node(state)

        assert "analysis" in result["completed_steps"]
        assert "summary" in result["analysis_results"]
        assert "Analysis complete" in result["analysis_results"]["summary"]

    def test_visualization_node_stores_results(self, agent):
        """Test that _visualization_node invokes the agent and stores results."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Chart generated successfully")]
        }

        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="Create a bar chart")],
            "completed_steps": [],
            "analysis_results": {},
            "visualization_results": {},
            "calculation_results": {},
            "pii_results": {},
        }

        with patch.object(
            type(agent),
            "_visualization_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._visualization_node(state)

        assert "visualization" in result["completed_steps"]
        assert "summary" in result["visualization_results"]

    def test_calculation_node_stores_results(self, agent):
        """Test that _calculation_node invokes the agent and stores results."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="Result: 42")]}

        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="Calculate 6 * 7")],
            "completed_steps": [],
            "analysis_results": {},
            "visualization_results": {},
            "calculation_results": {},
            "pii_results": {},
        }

        with patch.object(
            type(agent),
            "_calculation_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._calculation_node(state)

        assert "calculation" in result["completed_steps"]
        assert "summary" in result["calculation_results"]
        assert "42" in result["calculation_results"]["summary"]

    def test_pii_node_stores_results(self, agent):
        """Test that _pii_node invokes the agent and stores results."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="PII redacted: 2 emails found")]
        }

        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="Remove PII from this text")],
            "completed_steps": [],
            "analysis_results": {},
            "visualization_results": {},
            "calculation_results": {},
            "pii_results": {},
        }

        with patch.object(
            type(agent),
            "_pii_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._pii_node(state)

        assert "pii" in result["completed_steps"]
        assert "summary" in result["pii_results"]

    # ------------------------------------------------------------------
    # Presenter node tests
    # ------------------------------------------------------------------

    def test_presenter_node_synthesises_results(self, agent):
        """Test that presenter_node calls LLM with synthesis prompt and returns empty dict.

        The presenter node relies on LangGraph's message streaming to deliver
        the response to the user via AG-UI events. It must NOT return messages
        in the state update to avoid duplicate TEXT_MESSAGE_START events.
        """
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Here are your results in a nice format."
        mock_llm.invoke.return_value = mock_response

        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [],
            "analysis_results": {"summary": "Data has 100 rows"},
            "visualization_results": {"summary": "Bar chart created"},
            "calculation_results": {},
            "pii_results": {},
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._presenter_node(state)

        # LLM must be called to produce the streaming response
        mock_llm.invoke.assert_called_once()
        # The synthesis prompt should mention the specialist results
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Data has 100 rows" in call_args
        assert "Bar chart created" in call_args
        # Must return empty dict to avoid duplicate AG-UI text message events
        assert result == {}

    def test_presenter_node_handles_conversational_message(self, agent):
        """Test that presenter_node responds conversationally when no specialists needed.

        The presenter node relies on LangGraph's message streaming to deliver
        the response to the user via AG-UI events. It must NOT return messages
        in the state update to avoid duplicate TEXT_MESSAGE_START events.
        """
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = (
            "Hello! I can help with data analysis, charts, math, and PII removal."
        )
        mock_llm.invoke.return_value = mock_response

        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="hello")],
            "analysis_results": {},
            "visualization_results": {},
            "calculation_results": {},
            "pii_results": {},
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._presenter_node(state)

        # LLM must be called to produce the streaming response
        mock_llm.invoke.assert_called_once()
        # The conversational prompt should include the user message
        call_args = mock_llm.invoke.call_args[0][0]
        assert "hello" in call_args
        # Must return empty dict to avoid duplicate AG-UI text message events
        assert result == {}

    # ------------------------------------------------------------------
    # Helper function tests
    # ------------------------------------------------------------------

    def test_last_ai_content_extracts_last_ai_message(self):
        """Test _last_ai_content extracts the last AI message."""
        result = {
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="first response"),
                AIMessage(content="second response"),
            ]
        }
        assert _last_ai_content(result) == "second response"

    def test_last_ai_content_returns_str_for_no_ai_messages(self):
        """Test _last_ai_content returns str(result) when no AI messages."""
        result = {"messages": [HumanMessage(content="hello")]}
        assert "hello" in _last_ai_content(result)

    def test_extract_latest_user_message(self):
        """Test _extract_latest_user_message gets the last human message."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "messages": [
                HumanMessage(content="first question"),
                AIMessage(content="response"),
                HumanMessage(content="second question"),
            ]
        }
        assert _extract_latest_user_message(state) == "second question"

    def test_extract_latest_user_message_empty(self):
        """Test _extract_latest_user_message returns empty string when no messages."""
        state: DataProcessingState = {"messages": []}  # type: ignore[assignment]
        assert _extract_latest_user_message(state) == ""

    def test_gather_prior_results_with_data(self):
        """Test _gather_prior_results collects available results."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "analysis_results": {"summary": "analysis done"},
            "visualization_results": {},
            "calculation_results": {"summary": "calc done"},
            "pii_results": {},
        }
        result = _gather_prior_results(state)
        assert "analysis done" in result
        assert "calc done" in result
        assert (
            "visualization" not in result.lower()
            or "Previous visualization" not in result
        )

    def test_gather_prior_results_empty(self):
        """Test _gather_prior_results returns empty string when no results."""
        state: DataProcessingState = {  # type: ignore[assignment]
            "analysis_results": {},
            "visualization_results": {},
            "calculation_results": {},
            "pii_results": {},
        }
        assert _gather_prior_results(state) == ""

    # ------------------------------------------------------------------
    # TaskClassification model tests
    # ------------------------------------------------------------------

    def test_task_classification_model(self):
        """Test TaskClassification pydantic model."""
        tc = TaskClassification(
            needs_analysis=True,
            needs_visualization=False,
            needs_calculation=True,
            needs_pii_removal=False,
        )
        assert tc.needs_analysis is True
        assert tc.needs_visualization is False
        assert tc.needs_calculation is True
        assert tc.needs_pii_removal is False
