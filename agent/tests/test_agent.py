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

import json
import os
from unittest.mock import ANY, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from agent import MyAgent
from agent.myagent import (
    INTAKE_CLASSIFICATION_PROMPT,
    MENU,
    OrderState,
    _extract_latest_user_message,
    _extract_order_summary_from_history,
    _extract_previous_order_from_history,
    _extract_tool_output,
    _extract_total_from_history,
    _has_pending_order,
    _last_ai_content,
    _route_after_intake,
    _route_after_validation,
)
from agent.tools import (
    calculate_order_price,
    confirm_order,
    extract_order_items,
    format_order_response,
    validate_order,
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

    # ------------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # LLM tests
    # ------------------------------------------------------------------

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
        """Test that tools returns all five order-processing tools."""
        tools = agent.tools
        assert len(tools) == 5
        tool_names = {t.name for t in tools}
        assert tool_names == {
            "extract_order_items",
            "validate_order",
            "calculate_order_price",
            "confirm_order",
            "format_order_response",
        }

    # ------------------------------------------------------------------
    # Menu constant test
    # ------------------------------------------------------------------

    def test_menu_constant(self):
        """Test that the MENU constant has the expected items and prices."""
        assert MENU == {"pizza": 10, "burger": 8, "coke": 3}

    # ------------------------------------------------------------------
    # Intake classification prompt test
    # ------------------------------------------------------------------

    def test_intake_classification_prompt_exists(self):
        """Test that the intake classification prompt is defined and non-empty."""
        assert INTAKE_CLASSIFICATION_PROMPT
        assert "new_order" in INTAKE_CLASSIFICATION_PROMPT
        assert "modification" in INTAKE_CLASSIFICATION_PROMPT
        assert "confirmation" in INTAKE_CLASSIFICATION_PROMPT
        assert "cancellation" in INTAKE_CLASSIFICATION_PROMPT
        assert "greeting" in INTAKE_CLASSIFICATION_PROMPT

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
                # Verify all 6 nodes are present (intake + 5 agent nodes)
                expected_nodes = {
                    "intake_node",
                    "extraction_node",
                    "validation_node",
                    "pricing_node",
                    "confirmation_node",
                    "final_response_node",
                }
                assert expected_nodes == set(workflow.nodes.keys())

    def test_prompt_template_property(self, agent):
        """Test that prompt_template returns a ChatPromptTemplate."""
        template = agent.prompt_template
        assert template is not None
        assert isinstance(template, ChatPromptTemplate)

    def test_prompt_template_contains_menu(self, agent):
        """Test that the prompt template mentions the menu items."""
        template = agent.prompt_template
        messages = template.invoke({"user_prompt_content": "test"}).to_messages()
        system_msg = messages[0].content
        assert "pizza" in system_msg.lower()
        assert "burger" in system_msg.lower()
        assert "coke" in system_msg.lower()

    # ------------------------------------------------------------------
    # Route after validation tests
    # ------------------------------------------------------------------

    def test_route_after_validation_valid(self):
        """Test routing to pricing_node when order is valid."""
        state: OrderState = {"is_valid": True}  # type: ignore[assignment]
        assert _route_after_validation(state) == "pricing_node"

    def test_route_after_validation_invalid(self):
        """Test routing to final_response_node when order is invalid."""
        state: OrderState = {"is_valid": False}  # type: ignore[assignment]
        assert _route_after_validation(state) == "final_response_node"

    def test_route_after_validation_missing_defaults_to_invalid(self):
        """Test routing defaults to final_response_node when is_valid is missing."""
        state: OrderState = {}  # type: ignore[assignment]
        assert _route_after_validation(state) == "final_response_node"

    # ------------------------------------------------------------------
    # Route after intake tests
    # ------------------------------------------------------------------

    def test_route_after_intake_new_order(self):
        """Test routing to extraction_node when intent is new_order."""
        state: OrderState = {  # type: ignore[assignment]
            "intent": "new_order",
            "has_order_intent": True,
            "is_confirmation_reply": False,
        }
        assert _route_after_intake(state) == "extraction_node"

    def test_route_after_intake_modification(self):
        """Test routing to extraction_node when intent is modification."""
        state: OrderState = {  # type: ignore[assignment]
            "intent": "modification",
            "has_order_intent": True,
            "is_confirmation_reply": False,
        }
        assert _route_after_intake(state) == "extraction_node"

    def test_route_after_intake_confirmation(self):
        """Test routing to confirmation_node when intent is confirmation."""
        state: OrderState = {  # type: ignore[assignment]
            "intent": "confirmation",
            "has_order_intent": False,
            "is_confirmation_reply": True,
        }
        assert _route_after_intake(state) == "confirmation_node"

    def test_route_after_intake_cancellation(self):
        """Test routing to final_response_node when intent is cancellation."""
        state: OrderState = {  # type: ignore[assignment]
            "intent": "cancellation",
            "has_order_intent": False,
            "is_confirmation_reply": False,
        }
        assert _route_after_intake(state) == "final_response_node"

    def test_route_after_intake_greeting(self):
        """Test routing to final_response_node when intent is greeting."""
        state: OrderState = {  # type: ignore[assignment]
            "intent": "greeting",
            "has_order_intent": False,
            "is_confirmation_reply": False,
        }
        assert _route_after_intake(state) == "final_response_node"

    def test_route_after_intake_missing_defaults_to_greeting(self):
        """Test routing defaults to final_response_node when fields are missing."""
        state: OrderState = {}  # type: ignore[assignment]
        assert _route_after_intake(state) == "final_response_node"

    # ------------------------------------------------------------------
    # Intake node tests (LLM-based)
    # ------------------------------------------------------------------

    def test_intake_node_classifies_new_order(self, agent):
        """Test that intake_node classifies food order as new_order."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "new_order"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want 2 pizzas")],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "new_order"
        assert result["has_order_intent"] is True
        assert result["is_confirmation_reply"] is False
        assert result["is_modification"] is False
        assert "intake" in result["completed_steps"]

    def test_intake_node_classifies_greeting(self, agent):
        """Test that intake_node classifies greeting correctly."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "greeting"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="Hello there!")],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "greeting"
        assert result["has_order_intent"] is False
        assert result["is_confirmation_reply"] is False

    def test_intake_node_classifies_modification(self, agent):
        """Test that intake_node classifies 'no, add 3 burgers' as modification."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "modification"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="🧾 Order summary:\n"
                    "🍕 pizza — 2 x 10 dollars each = 20 dollars\n"
                    "💳 Grand total: 20 dollars\n"
                    "Would you like to confirm this order? Reply yes to confirm."
                ),
                HumanMessage(content="No, add 3 burgers please"),
            ],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "modification"
        assert result["has_order_intent"] is True
        assert result["is_modification"] is True
        assert result["is_confirmation_reply"] is False
        assert result["is_cancellation"] is False

    def test_intake_node_classifies_confirmation(self, agent):
        """Test that intake_node classifies 'yes' as confirmation when pending order."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "confirmation"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="Would you like to confirm this order? Reply yes to confirm."
                ),
                HumanMessage(content="yes"),
            ],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "confirmation"
        assert result["is_confirmation_reply"] is True

    def test_intake_node_classifies_cancellation(self, agent):
        """Test that intake_node classifies bare 'no' as cancellation when pending order."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "cancellation"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="Would you like to confirm this order? Reply yes to confirm."
                ),
                HumanMessage(content="no"),
            ],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "cancellation"
        assert result["is_cancellation"] is True
        assert result["is_confirmation_reply"] is False
        assert result["has_order_intent"] is False

    def test_intake_node_confirmation_without_pending_order_becomes_greeting(
        self, agent
    ):
        """Test that 'yes' without a pending order is treated as greeting."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "confirmation"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="yes")],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        # No pending order, so confirmation becomes greeting
        assert result["intent"] == "greeting"
        assert result["is_confirmation_reply"] is False

    def test_intake_node_cancellation_without_pending_order_becomes_greeting(
        self, agent
    ):
        """Test that 'cancel' without a pending order is treated as greeting."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "cancellation"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="cancel")],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "greeting"
        assert result["is_cancellation"] is False

    def test_intake_node_handles_unexpected_llm_output(self, agent):
        """Test that intake_node handles unexpected LLM output gracefully."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "I think this is a new_order request"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want pizza")],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        # Should parse "new_order" from the response
        assert result["intent"] == "new_order"

    def test_intake_node_completely_unexpected_llm_output_defaults_to_greeting(
        self, agent
    ):
        """Test that completely unexpected LLM output defaults to greeting."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "banana smoothie"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="hello")],
            "completed_steps": [],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._intake_node(state)

        assert result["intent"] == "greeting"

    # ------------------------------------------------------------------
    # Sub-agent factory tests
    # ------------------------------------------------------------------

    @patch("agent.myagent.create_agent")
    def test_extraction_agent_created_with_correct_tool(self, mock_create_agent, agent):
        """Test that _extraction_agent is created with extract_order_items tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._extraction_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "extract_order_items" in tool_names
                assert kwargs["name"] == "extraction_agent"

    @patch("agent.myagent.create_agent")
    def test_modification_extraction_agent_created_with_correct_tool(
        self, mock_create_agent, agent
    ):
        """Test that _modification_extraction_agent is created with extract_order_items tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._modification_extraction_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "extract_order_items" in tool_names
                assert kwargs["name"] == "extraction_agent"

    @patch("agent.myagent.create_agent")
    def test_validation_agent_created_with_correct_tool(self, mock_create_agent, agent):
        """Test that _validation_agent is created with validate_order tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._validation_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "validate_order" in tool_names
                assert kwargs["name"] == "validation_agent"

    @patch("agent.myagent.create_agent")
    def test_pricing_agent_created_with_correct_tool(self, mock_create_agent, agent):
        """Test that _pricing_agent is created with calculate_order_price tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._pricing_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "calculate_order_price" in tool_names
                assert kwargs["name"] == "pricing_agent"

    @patch("agent.myagent.create_agent")
    def test_confirmation_agent_created_with_correct_tool(
        self, mock_create_agent, agent
    ):
        """Test that _confirmation_agent is created with confirm_order tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._confirmation_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "confirm_order" in tool_names
                assert kwargs["name"] == "confirmation_agent"

    @patch("agent.myagent.create_agent")
    def test_final_response_agent_created_with_correct_tool(
        self, mock_create_agent, agent
    ):
        """Test that _final_response_agent is created with format_order_response tool."""
        mock_llm = Mock()
        with patch.object(agent, "llm", return_value=mock_llm):
            with patch.object(
                type(agent), "mcp_tools", new_callable=lambda: property(lambda self: [])
            ):
                _ = agent._final_response_agent
                mock_create_agent.assert_called_once()
                _, kwargs = mock_create_agent.call_args
                tool_names = {t.name for t in kwargs["tools"]}
                assert "format_order_response" in tool_names
                assert kwargs["name"] == "final_response_agent"

    # ------------------------------------------------------------------
    # Node wrapper tests
    # ------------------------------------------------------------------

    def test_extraction_node_stores_results(self, agent):
        """Test that _extraction_node invokes the sub-agent and stores extracted items."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    content='{"items": [{"item": "pizza", "quantity": 2}]}',
                    name="extract_order_items",
                    tool_call_id="1",
                ),
                AIMessage(content="I found 2 pizzas in your order."),
            ]
        }

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want 2 pizzas")],
            "completed_steps": [],
            "extracted_items": "",
            "is_modification": False,
        }

        with patch.object(
            type(agent),
            "_extraction_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._extraction_node(state)

        assert "extraction" in result["completed_steps"]
        assert "extracted_items" in result
        data = json.loads(result["extracted_items"])
        assert "items" in data

    def test_extraction_node_modification_uses_modification_agent(self, agent):
        """Test that _extraction_node uses the modification agent for modifications."""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    content='{"items": [{"item": "pizza", "quantity": 2}, {"item": "burger", "quantity": 3}]}',
                    name="extract_order_items",
                    tool_call_id="1",
                ),
                AIMessage(content="Updated order: 2 pizzas and 3 burgers."),
            ]
        }

        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="🧾 Order summary:\n"
                    "🍕 pizza — 2 x 10 dollars each = 20 dollars\n"
                    "💳 Grand total: 20 dollars\n"
                    "Would you like to confirm this order? Reply yes to confirm."
                ),
                HumanMessage(content="No, add 3 burgers please"),
            ],
            "completed_steps": [],
            "extracted_items": "",
            "is_modification": True,
        }

        with patch.object(
            type(agent),
            "_modification_extraction_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._extraction_node(state)

        assert "extraction" in result["completed_steps"]
        data = json.loads(result["extracted_items"])
        assert len(data["items"]) == 2

    def test_validation_node_stores_results_valid(self, agent):
        """Test that _validation_node correctly identifies a valid order."""
        valid_json = json.dumps(
            {
                "is_valid": True,
                "items": [{"item": "pizza", "quantity": 2}],
                "errors": [],
            }
        )
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    content=valid_json, name="validate_order", tool_call_id="1"
                ),
                AIMessage(content="All items are valid!"),
            ]
        }

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want 2 pizzas")],
            "completed_steps": ["extraction"],
            "extracted_items": '{"items": [{"item": "pizza", "quantity": 2}]}',
        }

        with patch.object(
            type(agent),
            "_validation_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._validation_node(state)

        assert "validation" in result["completed_steps"]
        assert result["is_valid"] is True

    def test_validation_node_stores_results_invalid(self, agent):
        """Test that _validation_node correctly identifies an invalid order."""
        invalid_json = json.dumps(
            {
                "is_valid": False,
                "items": [{"item": "sushi", "quantity": 1}],
                "errors": ["'sushi' is not on the menu."],
            }
        )
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    content=invalid_json, name="validate_order", tool_call_id="1"
                ),
                AIMessage(content="Sorry, sushi is not on our menu."),
            ]
        }

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want sushi")],
            "completed_steps": ["extraction"],
            "extracted_items": '{"items": [{"item": "sushi", "quantity": 1}]}',
        }

        with patch.object(
            type(agent),
            "_validation_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._validation_node(state)

        assert "validation" in result["completed_steps"]
        assert result["is_valid"] is False

    def test_pricing_node_stores_results(self, agent):
        """Test that _pricing_node invokes the sub-agent and stores pricing data."""
        pricing_json = json.dumps(
            {
                "line_items": [
                    {"item": "pizza", "quantity": 2, "unit_price": 10, "line_total": 20}
                ],
                "grand_total": 20,
            }
        )
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [
                ToolMessage(
                    content=pricing_json, name="calculate_order_price", tool_call_id="1"
                ),
                AIMessage(content="2x pizza = 20 dollars. Total: 20 dollars."),
            ]
        }

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want 2 pizzas")],
            "completed_steps": ["extraction", "validation"],
            "extracted_items": '{"items": [{"item": "pizza", "quantity": 2}]}',
        }

        with patch.object(
            type(agent),
            "_pricing_agent",
            new_callable=lambda: property(lambda self: mock_agent),
        ):
            result = agent._pricing_node(state)

        assert "pricing" in result["completed_steps"]
        assert "pricing_result" in result

    def test_confirmation_node_confirms(self, agent):
        """Test that _confirmation_node builds a confirmed result."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="🧾 Order Summary:\n- pizza 2x = 20 dollars\nTotal: 59 dollars\nWould you like to confirm? Reply yes to confirm."
                ),
                HumanMessage(content="yes"),
            ],
            "completed_steps": ["intake"],
        }

        result = agent._confirmation_node(state)

        assert "confirmation" in result["completed_steps"]
        assert result["is_valid"] is True
        data = json.loads(result["confirmation_result"])
        assert data["confirmed"] is True
        assert data["grand_total"] == 59

    # ------------------------------------------------------------------
    # Final response node tests
    # ------------------------------------------------------------------

    def test_final_response_node_asks_for_confirmation(self, agent):
        """Test final_response_node asks for confirmation after pricing."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Would you like to confirm? Reply yes."
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want 2 pizzas")],
            "is_valid": True,
            "has_order_intent": True,
            "is_confirmation_reply": False,
            "is_cancellation": False,
            "intent": "new_order",
            "pricing_result": json.dumps({"grand_total": 20}),
            "completed_steps": ["extraction", "validation", "pricing"],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._final_response_node(state)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "confirm" in call_args.lower()
        assert result == {}

    def test_final_response_node_confirmed_order(self, agent):
        """Test final_response_node with a confirmed order."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Your order is confirmed! 🎉"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="yes")],
            "is_valid": True,
            "has_order_intent": False,
            "is_confirmation_reply": True,
            "is_cancellation": False,
            "intent": "confirmation",
            "confirmation_result": json.dumps(
                {"confirmed": True, "message": "Order confirmed!", "grand_total": 20}
            ),
            "completed_steps": ["intake", "confirmation"],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._final_response_node(state)

        mock_llm.invoke.assert_called_once()
        assert result == {}

    def test_final_response_node_cancelled_order(self, agent):
        """Test final_response_node with a cancelled order."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Order cancelled. Feel free to order again!"
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="no")],
            "is_valid": False,
            "has_order_intent": False,
            "is_confirmation_reply": False,
            "is_cancellation": True,
            "intent": "cancellation",
            "completed_steps": ["intake"],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._final_response_node(state)

        mock_llm.invoke.assert_called_once()
        assert result == {}

    def test_final_response_node_invalid_order(self, agent):
        """Test final_response_node with an invalid order."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Sorry, sushi is not on our menu."
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="I want sushi")],
            "is_valid": False,
            "has_order_intent": True,
            "is_confirmation_reply": False,
            "is_cancellation": False,
            "intent": "new_order",
            "validation_result": json.dumps(
                {"is_valid": False, "errors": ["'sushi' is not on the menu."]}
            ),
            "completed_steps": ["extraction", "validation"],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._final_response_node(state)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "sushi" in call_args.lower()
        assert result == {}

    def test_final_response_node_conversational(self, agent):
        """Test final_response_node with a conversational (non-order) message."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Hello! Welcome to our restaurant."
        mock_llm.invoke.return_value = mock_response

        state: OrderState = {  # type: ignore[assignment]
            "messages": [HumanMessage(content="hello")],
            "is_valid": False,
            "has_order_intent": False,
            "is_confirmation_reply": False,
            "is_cancellation": False,
            "intent": "greeting",
            "completed_steps": ["intake"],
        }

        with patch.object(agent, "llm", return_value=mock_llm):
            result = agent._final_response_node(state)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "hello" in call_args
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
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                HumanMessage(content="first question"),
                AIMessage(content="response"),
                HumanMessage(content="second question"),
            ]
        }
        assert _extract_latest_user_message(state) == "second question"

    def test_extract_latest_user_message_empty(self):
        """Test _extract_latest_user_message returns empty string when no messages."""
        state: OrderState = {"messages": []}  # type: ignore[assignment]
        assert _extract_latest_user_message(state) == ""

    def test_extract_tool_output_found(self):
        """Test _extract_tool_output finds the correct tool output."""
        result = {
            "messages": [
                HumanMessage(content="test"),
                ToolMessage(
                    content='{"items": []}',
                    name="extract_order_items",
                    tool_call_id="1",
                ),
                AIMessage(content="Done"),
            ]
        }
        output = _extract_tool_output(result, "extract_order_items")
        assert output == '{"items": []}'

    def test_extract_tool_output_not_found(self):
        """Test _extract_tool_output returns empty string when tool not found."""
        result = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="Done"),
            ]
        }
        output = _extract_tool_output(result, "extract_order_items")
        assert output == ""

    def test_has_pending_order_true(self):
        """Test _has_pending_order returns True when AI asked for order confirmation."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="Would you like to confirm this order? Reply yes to confirm."
                ),
                HumanMessage(content="yes"),
            ]
        }
        assert _has_pending_order(state) is True

    def test_has_pending_order_false(self):
        """Test _has_pending_order returns False when AI didn't ask."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content="Hello! How can I help you?"),
                HumanMessage(content="yes"),
            ]
        }
        assert _has_pending_order(state) is False

    def test_has_pending_order_false_for_menu_question(self):
        """Test _has_pending_order returns False for 'would you like to order?' messages."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="Sure — I can help! Our menu: Pizza: 10 dollars. What would you like to order?"
                ),
                HumanMessage(content="I want 2 pizzas with 15 cokes"),
            ]
        }
        assert _has_pending_order(state) is False

    def test_has_pending_order_empty(self):
        """Test _has_pending_order returns False with no messages."""
        state: OrderState = {"messages": []}  # type: ignore[assignment]
        assert _has_pending_order(state) is False

    def test_has_pending_order_skips_system_messages(self):
        """Test _has_pending_order skips SystemMessages between AI and Human."""
        from langchain_core.messages import SystemMessage

        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content="Would you like to confirm? Reply yes to confirm."),
                SystemMessage(content="You are a Smart Order Assistant."),
                HumanMessage(content="yes"),
            ]
        }
        assert _has_pending_order(state) is True

    def test_has_pending_order_only_human_message(self):
        """Test _has_pending_order returns False with only a HumanMessage."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                HumanMessage(content="yes"),
            ]
        }
        assert _has_pending_order(state) is False

    def test_extract_total_from_history_found(self):
        """Test _extract_total_from_history finds total from human-readable text."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="2x pizza = 20 dollars\n3x burger = 24 dollars\nTotal: 59 dollars"
                ),
            ]
        }
        assert _extract_total_from_history(state) == 59

    def test_extract_total_from_history_grand_total(self):
        """Test _extract_total_from_history finds 'Grand Total' pattern."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(
                    content="💰 Grand Total: 42 dollars\nWould you like to confirm?"
                ),
            ]
        }
        assert _extract_total_from_history(state) == 42

    def test_extract_total_from_history_not_found(self):
        """Test _extract_total_from_history returns 0 when no total found."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content="Hello! How can I help you?"),
            ]
        }
        assert _extract_total_from_history(state) == 0

    def test_extract_order_summary_from_history_found(self):
        """Test _extract_order_summary_from_history finds the summary message."""
        summary = "🧾 Order Summary:\n- pizza 2x\nTotal: 20 dollars\nWould you like to confirm?"
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content="Hello!"),
                AIMessage(content=summary),
            ]
        }
        assert _extract_order_summary_from_history(state) == summary

    def test_extract_order_summary_from_history_not_found(self):
        """Test _extract_order_summary_from_history returns empty when no summary."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content="Hello! How can I help you?"),
            ]
        }
        assert _extract_order_summary_from_history(state) == ""

    def test_extract_previous_order_from_history_found(self):
        """Test _extract_previous_order_from_history finds previous order details."""
        order_msg = "🧾 Order summary:\n🍕 pizza — 2 x 10 dollars each = 20 dollars\n💳 Grand total: 20 dollars"
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content=order_msg),
                HumanMessage(content="add 3 burgers"),
            ]
        }
        assert _extract_previous_order_from_history(state) == order_msg

    def test_extract_previous_order_from_history_not_found(self):
        """Test _extract_previous_order_from_history returns fallback when no order."""
        state: OrderState = {  # type: ignore[assignment]
            "messages": [
                AIMessage(content="Hello! How can I help?"),
            ]
        }
        assert _extract_previous_order_from_history(state) == "No previous order found."

    # ------------------------------------------------------------------
    # Direct tool tests
    # ------------------------------------------------------------------

    def test_extract_order_items_tool_with_quantities(self):
        """Test extract_order_items extracts items with quantities."""
        result = extract_order_items.invoke(
            {"user_input": "I want 2 pizzas and 3 cokes"}
        )
        data = json.loads(result)
        assert len(data["items"]) == 2
        items_dict = {i["item"]: i["quantity"] for i in data["items"]}
        assert items_dict["pizza"] == 2
        assert items_dict["coke"] == 3

    def test_extract_order_items_tool_without_quantities(self):
        """Test extract_order_items defaults to quantity 1."""
        result = extract_order_items.invoke({"user_input": "I want a burger"})
        data = json.loads(result)
        assert len(data["items"]) == 1
        assert data["items"][0]["item"] == "burger"
        assert data["items"][0]["quantity"] == 1

    def test_extract_order_items_tool_no_items(self):
        """Test extract_order_items with no menu items in input."""
        result = extract_order_items.invoke({"user_input": "hello there"})
        data = json.loads(result)
        assert len(data["items"]) == 0
        assert "error" in data

    def test_validate_order_tool_valid(self):
        """Test validate_order with valid items."""
        input_json = json.dumps({"items": [{"item": "pizza", "quantity": 2}]})
        result = validate_order.invoke({"extracted_items_json": input_json})
        data = json.loads(result)
        assert data["is_valid"] is True
        assert len(data["errors"]) == 0

    def test_validate_order_tool_invalid_item(self):
        """Test validate_order with an item not on the menu."""
        input_json = json.dumps({"items": [{"item": "sushi", "quantity": 1}]})
        result = validate_order.invoke({"extracted_items_json": input_json})
        data = json.loads(result)
        assert data["is_valid"] is False
        assert any("sushi" in e for e in data["errors"])

    def test_validate_order_tool_quantity_exceeds_max(self):
        """Test validate_order with quantity exceeding 10."""
        input_json = json.dumps({"items": [{"item": "pizza", "quantity": 15}]})
        result = validate_order.invoke({"extracted_items_json": input_json})
        data = json.loads(result)
        assert data["is_valid"] is False
        assert any("exceeds" in e for e in data["errors"])

    def test_validate_order_tool_empty_items(self):
        """Test validate_order with no items."""
        input_json = json.dumps({"items": []})
        result = validate_order.invoke({"extracted_items_json": input_json})
        data = json.loads(result)
        assert data["is_valid"] is False

    def test_calculate_order_price_tool(self):
        """Test calculate_order_price calculates correctly."""
        input_json = json.dumps(
            {
                "items": [
                    {"item": "pizza", "quantity": 2},
                    {"item": "coke", "quantity": 1},
                ]
            }
        )
        result = calculate_order_price.invoke({"validated_items_json": input_json})
        data = json.loads(result)
        assert data["grand_total"] == 23  # 2*10 + 1*3
        assert len(data["line_items"]) == 2

    def test_confirm_order_tool(self):
        """Test confirm_order auto-approves valid orders."""
        input_json = json.dumps(
            {
                "line_items": [
                    {"item": "pizza", "quantity": 2, "unit_price": 10, "line_total": 20}
                ],
                "grand_total": 20,
            }
        )
        result = confirm_order.invoke({"pricing_json": input_json})
        data = json.loads(result)
        assert data["confirmed"] is True
        assert data["grand_total"] == 20

    def test_format_order_response_tool_valid(self):
        """Test format_order_response with a confirmed order."""
        input_json = json.dumps(
            {
                "confirmed": True,
                "message": "Order confirmed!",
                "grand_total": 20,
            }
        )
        result = format_order_response.invoke({"order_data_json": input_json})
        assert "🎉" in result
        assert "Order confirmed" in result

    def test_format_order_response_tool_invalid(self):
        """Test format_order_response with an invalid order."""
        input_json = json.dumps(
            {
                "is_valid": False,
                "errors": ["'sushi' is not on the menu."],
            }
        )
        result = format_order_response.invoke({"order_data_json": input_json})
        assert "⚠️" in result
        assert "sushi" in result
