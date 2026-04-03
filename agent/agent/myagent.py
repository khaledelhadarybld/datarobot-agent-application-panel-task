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
"""Smart Order Assistant — multi-agent food ordering workflow.

A LangGraph workflow with 6 nodes that process a food order:

  0. intake_node         : LLM-based intent classification (order, modification,
                           confirmation, cancellation, or greeting).
  1. extraction_node     : Extracts items and quantities from the user's message.
                           For modifications, merges with the previous order.
  2. validation_node     : Checks items exist on the menu and quantity <= 10.
  3. pricing_node        : Calculates the total price (only if validation passes).
  4. confirmation_node   : Processes explicit user confirmation → finalizes order.
  5. final_response_node : Returns a friendly confirmation, summary, or error message.

Conditional routing:
  - If intake detects a greeting → final_response_node (conversational reply)
  - If intake detects a new order or modification → extraction_node
  - If intake detects an explicit confirmation → confirmation_node
  - If intake detects a cancellation → final_response_node (cancellation message)
  - If validation fails → skip pricing → final_response_node (show errors)
  - After pricing → final_response_node (ask user to confirm)
  - After confirmation → final_response_node (show confirmed order)

Example menu:
  - pizza  = $10
  - burger = $8
  - coke   = $3
"""

import json
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

from agent.config import Config
from agent.tools import (
    calculate_order_price,
    confirm_order,
    extract_order_items,
    format_order_response,
    validate_order,
)

# ---------------------------------------------------------------------------
# Restaurant menu — used by tools and prompts
# ---------------------------------------------------------------------------
MENU = {"pizza": 10, "burger": 8, "coke": 3}

# ---------------------------------------------------------------------------
# Shared typed state
# ---------------------------------------------------------------------------


class OrderState(MessagesState):
    """Shared state passed between all nodes in the order workflow.

    Think of this as a "clipboard" that every agent can read from and write to.
    Each field stores the output of one step so the next step can use it.
    """

    # Output of the extraction agent (JSON string of items + quantities)
    extracted_items: str

    # Output of the validation agent (JSON string with is_valid + errors)
    validation_result: str

    # Whether the order passed validation (used for conditional routing)
    is_valid: bool

    # Output of the pricing agent (JSON string with line items + total)
    pricing_result: str

    # Output of the confirmation agent (JSON string with confirmed + message)
    confirmation_result: str

    # Tracks which steps have been completed
    completed_steps: list[str]

    # Whether the intake node detected an order intent in the user's message
    has_order_intent: bool

    # Whether the user is replying to a confirmation prompt (yes/no)
    is_confirmation_reply: bool

    # Whether the user wants to modify an existing order
    is_modification: bool

    # Whether the user wants to cancel the order
    is_cancellation: bool

    # The intent classified by the LLM intake node
    intent: str


# ---------------------------------------------------------------------------
# Intent classification prompt for the intake node
# ---------------------------------------------------------------------------
INTAKE_CLASSIFICATION_PROMPT = (
    "You are an intent classifier for a restaurant order system.\n"
    "Classify the customer's message into exactly ONE of these intents:\n\n"
    "- **new_order**: The customer wants to place a new food order "
    "(mentions menu items, quantities, or ordering phrases).\n"
    "- **modification**: The customer wants to CHANGE an existing order "
    "(add items, remove items, change quantities). This includes replies "
    "like 'no, add 3 burgers' or 'actually, change to 5 pizzas' or "
    "'can you add a coke?' when there is already a pending order.\n"
    "- **confirmation**: The customer is explicitly confirming the order "
    "with NO modifications (e.g., just 'yes', 'confirm', 'go ahead', "
    "'sure', 'yep', 'ok').\n"
    "- **cancellation**: The customer clearly wants to cancel the entire "
    "order with NO new items (e.g., just 'no', 'cancel', 'never mind').\n"
    "- **greeting**: General conversation, greeting, or question that is "
    "not about placing/modifying an order.\n\n"
    "CRITICAL RULES:\n"
    "- If the message contains BOTH a negative word (no, nah) AND mentions "
    "food items or modifications, classify as **modification**, NOT cancellation.\n"
    "- 'No, add 3 burgers' → modification\n"
    "- 'No, I want 5 pizzas instead' → modification\n"
    "- 'No thanks' → cancellation\n"
    "- 'No' (alone) → cancellation\n"
    "- 'Can you add a coke?' → modification (if there's an existing order)\n\n"
    "Respond with ONLY the intent name (one word): "
    "new_order, modification, confirmation, cancellation, or greeting.\n"
    "Do not add any other text."
)


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------


def _route_after_intake(
    state: OrderState,
) -> Literal["extraction_node", "confirmation_node", "final_response_node"]:
    """Decide where to go after the intake check.

    Routes based on LLM-classified intent:
      - confirmation → confirmation_node
      - new_order or modification → extraction_node
      - cancellation or greeting → final_response_node
    """
    intent = state.get("intent", "greeting")

    if intent == "confirmation" and state.get("is_confirmation_reply", False):
        return "confirmation_node"
    if intent in ("new_order", "modification"):
        return "extraction_node"
    return "final_response_node"


def _route_after_validation(
    state: OrderState,
) -> Literal["pricing_node", "final_response_node"]:
    """Decide where to go after validation.

    If the order is valid   → continue to pricing_node
    If the order is invalid → skip pricing, go to final_response_node
    """
    if state.get("is_valid", False):
        return "pricing_node"
    else:
        return "final_response_node"


# ---------------------------------------------------------------------------
# MyAgent — Smart Order Assistant
# ---------------------------------------------------------------------------


class MyAgent(LangGraphAgent):
    """Smart Order Assistant with 5 sub-agents.

    Processes food orders through extraction → validation → pricing →
    (ask for confirmation) → confirmation → final response, with conditional
    routing that skips pricing/confirmation when validation fails.

    The intake node uses LLM-based intent classification to correctly handle
    order modifications (e.g., "no, add 3 burgers") without keyword matching.
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
    # Tools — all 5 order-processing tools
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[BaseTool]:
        """Return the list of all tools available to the agent."""
        return [
            extract_order_items,
            validate_order,
            calculate_order_price,
            confirm_order,
            format_order_response,
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
                    "You are a Smart Order Assistant for a restaurant. "
                    "You help customers place food orders. "
                    "Our menu: pizza (10 dollars), burger (8 dollars), coke (3 dollars). "
                    "Maximum 10 of any item per order.",
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
        appropriate HumanMessage / AIMessage so that the extraction node
        sees the complete conversation history.
        """
        history_messages: list[Any] = []
        all_messages = list(run_agent_input.messages or [])

        # Prior messages become raw history entries.
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
    # Sub-agent factories (each agent gets its own tool + system prompt)
    # ------------------------------------------------------------------

    @property
    def _extraction_agent(self) -> Any:
        """Create the Extraction Agent.

        This agent's job is to read the user's message and pull out
        which menu items they want and how many of each.
        """
        return create_agent(
            self.llm(),
            tools=[extract_order_items] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Extraction Agent for a restaurant order system.\n\n"
                "Your ONLY job is to extract food items and quantities from the "
                "user's message using the extract_order_items tool.\n\n"
                f"Our menu: {MENU}\n\n"
                "IMPORTANT: Always call the extract_order_items tool first.\n"
                "After the tool returns, summarize what you found in a brief, "
                "friendly sentence like: 'I found 2 pizzas and 3 cokes in your order.'\n"
                "Do NOT output raw JSON to the user. Always write a human-friendly summary."
            ),
            name="extraction_agent",
        )

    @property
    def _modification_extraction_agent(self) -> Any:
        """Create the Modification Extraction Agent.

        Similar to the extraction agent, but specifically designed for order
        modifications. It computes the merged order totals first, then calls
        the extraction tool with a clean merged order string.
        """
        return create_agent(
            self.llm(),
            tools=[extract_order_items] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Extraction Agent for a restaurant order system.\n\n"
                "Your job is to handle ORDER MODIFICATIONS. You will receive the "
                "previous order and the customer's modification request.\n\n"
                f"Our menu: {MENU}\n\n"
                "CRITICAL STEPS — follow these exactly:\n"
                "1. Read the PREVIOUS order items and quantities.\n"
                "2. Read the customer's modification (add/remove/change items).\n"
                "3. Compute the MERGED order by combining previous items with changes:\n"
                "   - 'add 3 burgers' to existing 3 burgers = 6 burgers total\n"
                "   - 'add 1 coke' to existing 5 cokes = 6 cokes total\n"
                "   - 'remove coke' = remove coke from order\n"
                "   - New items not in previous order are simply added.\n"
                "4. Call extract_order_items with ONLY the FINAL merged order as a "
                "simple sentence. Example: 'I want 2 pizzas, 6 burgers, and 6 cokes'\n"
                "   Do NOT include the previous order text or modification text — "
                "   ONLY the final merged totals.\n\n"
                "After the tool returns, summarize the FULL updated order in a brief, "
                "friendly sentence like: 'Your updated order: 2 pizzas, 6 burgers, and 6 cokes.'\n"
                "Do NOT output raw JSON to the user."
            ),
            name="extraction_agent",
        )

    @property
    def _validation_agent(self) -> Any:
        """Create the Validation Agent.

        This agent checks that every item exists on the menu and that
        no quantity exceeds 10 (our guardrail).
        """
        return create_agent(
            self.llm(),
            tools=[validate_order] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Validation Agent for a restaurant order system.\n\n"
                "Your ONLY job is to validate extracted order items using the "
                "validate_order tool.\n\n"
                "Guardrails:\n"
                f"- Items must be on the menu: {list(MENU.keys())}\n"
                "- Quantity per item must be between 1 and 10\n\n"
                "IMPORTANT: Always call the validate_order tool first.\n"
                "After the tool returns, summarize the result in a brief, "
                "friendly sentence like: 'All items are valid and on the menu!' "
                "or 'Sorry, sushi is not on our menu.'\n"
                "Do NOT output raw JSON to the user."
            ),
            name="validation_agent",
        )

    @property
    def _pricing_agent(self) -> Any:
        """Create the Pricing Agent.

        This agent calculates the price for each line item and the
        grand total using menu prices.
        """
        return create_agent(
            self.llm(),
            tools=[calculate_order_price] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Pricing Agent for a restaurant order system.\n\n"
                "Your ONLY job is to calculate the total price using the "
                "calculate_order_price tool.\n\n"
                f"Menu prices: {MENU}\n\n"
                "IMPORTANT: Always call the calculate_order_price tool first.\n"
                "After the tool returns, present the pricing as a clear summary like:\n"
                "  2x pizza at 10 dollars each = 20 dollars\n"
                "  3x coke at 3 dollars each = 9 dollars\n"
                "  Total: 29 dollars\n\n"
                "Do NOT output raw JSON. Write prices as 'X dollars' not with dollar signs."
            ),
            name="pricing_agent",
        )

    @property
    def _confirmation_agent(self) -> Any:
        """Create the Confirmation Agent.

        This agent processes the user's explicit confirmation to finalize.
        """
        return create_agent(
            self.llm(),
            tools=[confirm_order] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Confirmation Agent for a restaurant order system.\n\n"
                "Your ONLY job is to confirm the order using the "
                "confirm_order tool.\n\n"
                "You will receive the pricing data. Call the confirm_order tool "
                "with it to finalize the order.\n"
                "After the tool returns, say something like: "
                "'Your order has been confirmed!'\n"
                "Do NOT output raw JSON."
            ),
            name="confirmation_agent",
        )

    @property
    def _final_response_agent(self) -> Any:
        """Create the Final Response Agent.

        This agent formats the final message shown to the user — either
        a confirmation or an error message.
        """
        return create_agent(
            self.llm(),
            tools=[format_order_response] + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are the Final Response Agent for a restaurant order system.\n\n"
                "Your ONLY job is to format the final response using the "
                "format_order_response tool.\n\n"
                "For valid orders: pass the confirmation data.\n"
                "For invalid orders: pass the validation errors.\n\n"
                "Always call the format_order_response tool. "
                "Return a friendly, clear message for the customer.\n"
                "Do NOT output raw JSON."
            ),
            name="final_response_agent",
        )

    # ------------------------------------------------------------------
    # Node wrappers — each node runs its sub-agent and stores results
    # ------------------------------------------------------------------

    def _intake_node(self, state: OrderState) -> dict[str, Any]:
        """Step 0: LLM-based intent classification.

        Uses the LLM to classify the user's message into one of:
          - new_order: placing a new food order
          - modification: modifying an existing pending order
          - confirmation: explicitly confirming a pending order
          - cancellation: explicitly cancelling with no modifications
          - greeting: general conversation

        This replaces brittle keyword matching and correctly handles
        nuanced messages like "no, add 3 burgers please".
        """
        user_message = _extract_latest_user_message(state)

        # Build context about whether there's a pending order
        has_pending_order = _has_pending_order(state)
        context = ""
        if has_pending_order:
            context = (
                "\n\nCONTEXT: There is a pending order that the customer was asked "
                "to confirm. The customer's reply may be a confirmation, cancellation, "
                "or a request to modify the order."
            )

        # Use the LLM to classify intent
        classification_prompt = (
            f"{INTAKE_CLASSIFICATION_PROMPT}{context}\n\n"
            f'Customer message: "{user_message}"'
        )

        llm = self.llm()
        response = llm.invoke(classification_prompt)
        intent_raw = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        intent = intent_raw.strip().lower()

        # Normalize the intent to one of the known values
        valid_intents = {
            "new_order",
            "modification",
            "confirmation",
            "cancellation",
            "greeting",
        }
        if intent not in valid_intents:
            # Fallback: if the LLM returned something unexpected, try to parse it
            for valid in valid_intents:
                if valid in intent:
                    intent = valid
                    break
            else:
                intent = "greeting"

        # Map intent to state flags
        has_order_intent = intent in ("new_order", "modification")
        is_confirmation_reply = intent == "confirmation" and has_pending_order
        is_modification = intent == "modification"
        is_cancellation = intent == "cancellation"

        # If classified as confirmation but no pending order, treat as greeting
        if intent == "confirmation" and not has_pending_order:
            is_confirmation_reply = False
            has_order_intent = False
            intent = "greeting"

        # If classified as cancellation but no pending order, treat as greeting
        if intent == "cancellation" and not has_pending_order:
            is_cancellation = False
            intent = "greeting"

        if self.verbose:
            print(
                f"[intake_node] message='{user_message}', "
                f"intent={intent}, "
                f"has_order_intent={has_order_intent}, "
                f"is_confirmation_reply={is_confirmation_reply}, "
                f"is_modification={is_modification}, "
                f"is_cancellation={is_cancellation}, "
                f"has_pending_order={has_pending_order}"
            )

        completed = list(state.get("completed_steps", []))
        if "intake" not in completed:
            completed.append("intake")

        return {
            "intent": intent,
            "has_order_intent": has_order_intent,
            "is_confirmation_reply": is_confirmation_reply,
            "is_modification": is_modification,
            "is_cancellation": is_cancellation,
            "completed_steps": completed,
        }

    def _extraction_node(self, state: OrderState) -> dict[str, Any]:
        """Step 1: Extract items and quantities from the user's order message.

        For new orders: extracts items from the current message.
        For modifications: builds the complete updated order by combining
        previous items from conversation history with the requested changes,
        then extracts items from that combined description.
        """
        user_message = _extract_latest_user_message(state)
        is_modification = state.get("is_modification", False)

        if self.verbose:
            print(
                f"[extraction_node] Input: {user_message}, is_modification={is_modification}"
            )

        if is_modification:
            # Build the full updated order from conversation history + modification
            previous_order = _extract_previous_order_from_history(state)
            combined_order = (
                f"The customer's PREVIOUS order was:\n{previous_order}\n\n"
                f'The customer now says: "{user_message}"\n\n'
                "INSTRUCTIONS:\n"
                "1. First, compute the MERGED order totals by combining the previous "
                "order with the customer's changes. For example:\n"
                "   - Previous: 3 burgers, 5 cokes. Customer: 'add 3 burgers and 1 coke'\n"
                "   - Merged: 6 burgers, 6 cokes\n"
                "2. Then call extract_order_items with ONLY the final merged totals "
                "as a simple order sentence. Example: 'I want 6 burgers and 6 cokes'\n"
                "   Do NOT pass the previous order text or modification request to the tool."
            )
            context_msg = HumanMessage(content=combined_order)
            result = self._modification_extraction_agent.invoke(
                {"messages": [context_msg]}
            )
        else:
            # New order — extract from current message
            context_msg = HumanMessage(
                content=(
                    f"The customer said: {user_message}\n\n"
                    "Please extract the food items and quantities from this message "
                    "using the extract_order_items tool. After calling the tool, "
                    "summarize what you found in a brief friendly sentence."
                )
            )
            result = self._extraction_agent.invoke({"messages": [context_msg]})

        # Get the raw tool output (JSON) from the result
        extracted_json = _extract_tool_output(result, "extract_order_items")
        if not extracted_json:
            extracted_json = _last_ai_content(result)

        if self.verbose:
            print(f"[extraction_node] Output: {extracted_json}")

        completed = list(state.get("completed_steps", []))
        if "extraction" not in completed:
            completed.append("extraction")

        return {
            "extracted_items": extracted_json,
            "completed_steps": completed,
        }

    def _validation_node(self, state: OrderState) -> dict[str, Any]:
        """Step 2: Validate the extracted items against the menu and quantity limits.

        Uses the validation sub-agent (LLM + tool) so the tool call is
        visible in the UI.
        """
        extracted_items = state.get("extracted_items", "{}")

        if self.verbose:
            print(f"[validation_node] Input: {extracted_items}")

        # Ask the validation agent to check the order
        context_msg = HumanMessage(
            content=(
                f"Please validate this order using the validate_order tool:\n"
                f"{extracted_items}\n\n"
                "After calling the tool, summarize whether the order is valid "
                "in a brief friendly sentence."
            )
        )
        result = self._validation_agent.invoke({"messages": [context_msg]})

        # Get the raw tool output
        validation_json = _extract_tool_output(result, "validate_order")
        if not validation_json:
            validation_json = _last_ai_content(result)

        # Parse is_valid from the validation result
        is_valid = False
        try:
            validation_data = json.loads(validation_json)
            is_valid = validation_data.get("is_valid", False)
        except (json.JSONDecodeError, TypeError):
            pass

        if self.verbose:
            print(f"[validation_node] is_valid = {is_valid}")

        completed = list(state.get("completed_steps", []))
        if "validation" not in completed:
            completed.append("validation")

        return {
            "validation_result": validation_json,
            "is_valid": is_valid,
            "completed_steps": completed,
        }

    def _pricing_node(self, state: OrderState) -> dict[str, Any]:
        """Step 3: Calculate the total price (only reached if validation passed).

        Uses the pricing sub-agent (LLM + tool) so the tool call is
        visible in the UI.
        """
        extracted_items = state.get("extracted_items", "{}")

        if self.verbose:
            print(f"[pricing_node] Input: {extracted_items}")

        # Ask the pricing agent to calculate the total
        context_msg = HumanMessage(
            content=(
                f"Please calculate the price using the calculate_order_price tool:\n"
                f"{extracted_items}\n\n"
                "After calling the tool, present the pricing as a clear summary "
                "with line items and total. Write prices as 'X dollars'."
            )
        )
        result = self._pricing_agent.invoke({"messages": [context_msg]})

        pricing_json = _extract_tool_output(result, "calculate_order_price")
        if not pricing_json:
            pricing_json = _last_ai_content(result)

        if self.verbose:
            print(f"[pricing_node] Output: {pricing_json}")

        completed = list(state.get("completed_steps", []))
        if "pricing" not in completed:
            completed.append("pricing")

        return {
            "pricing_result": pricing_json,
            "completed_steps": completed,
        }

    def _confirmation_node(self, state: OrderState) -> dict[str, Any]:
        """Step 4: Process the user's explicit confirmation.

        This node is only reached when the LLM intake classified the
        message as an explicit confirmation. It extracts order details
        from conversation history and builds the confirmation result.
        """
        user_message = _extract_latest_user_message(state)

        if self.verbose:
            print(f"[confirmation_node] user_message='{user_message}'")

        # User confirmed — extract the grand total and order summary
        # from the conversation history (human-readable AI messages)
        grand_total = _extract_total_from_history(state)
        order_summary = _extract_order_summary_from_history(state)

        confirmation_json = json.dumps(
            {
                "confirmed": True,
                "message": "Order confirmed!",
                "grand_total": grand_total,
                "order_summary": order_summary,
            }
        )

        if self.verbose:
            print(f"[confirmation_node] Output: {confirmation_json}")

        completed = list(state.get("completed_steps", []))
        if "confirmation" not in completed:
            completed.append("confirmation")

        return {
            "confirmation_result": confirmation_json,
            "is_valid": True,
            "completed_steps": completed,
        }

    def _final_response_node(self, state: OrderState) -> dict[str, Any]:
        """Step 5: Format and return the final response to the customer.

        This is the terminal node. It uses the LLM to stream the response
        back to the user via AG-UI events.

        Handles 5 scenarios:
        - Greeting (no order): conversational reply with menu
        - Valid order (not yet confirmed): show summary + ask for confirmation
        - Confirmed order: show confirmation message
        - Cancelled order: polite cancellation message
        - Invalid order: show errors + menu
        """
        is_valid = state.get("is_valid", False)
        is_confirmation_reply = state.get("is_confirmation_reply", False)
        is_cancellation = state.get("is_cancellation", False)
        has_order_intent = state.get("has_order_intent", False)
        intent = state.get("intent", "greeting")

        if self.verbose:
            print(
                f"[final_response_node] intent={intent}, is_valid={is_valid}, "
                f"is_confirmation_reply={is_confirmation_reply}, "
                f"is_cancellation={is_cancellation}, "
                f"has_order_intent={has_order_intent}"
            )

        # Scenario 1: User just confirmed an order
        if is_confirmation_reply:
            confirmation_data = state.get("confirmation_result", "{}")
            try:
                conf = json.loads(confirmation_data)
                if conf.get("confirmed", False):
                    prompt = (
                        "You are a Smart Order Assistant. The customer just confirmed "
                        "their order. Present a cheerful confirmation message.\n\n"
                        "Rules:\n"
                        "- Use emojis and friendly language\n"
                        "- Show the order details if available\n"
                        "- Thank the customer\n"
                        "- Write prices as 'X dollars' not with dollar signs\n\n"
                        f"Confirmation data:\n{confirmation_data}"
                    )
                else:
                    prompt = (
                        "You are a Smart Order Assistant. Something went wrong. "
                        "Apologize and ask the customer to try again."
                    )
            except (json.JSONDecodeError, TypeError):
                prompt = (
                    "You are a Smart Order Assistant. Something went wrong with "
                    "the confirmation. Apologize and ask the customer to try again."
                )
            self.llm().invoke(prompt)
            return {}

        # Scenario 2: User cancelled the order
        if is_cancellation:
            prompt = (
                "You are a Smart Order Assistant. The customer cancelled "
                "their order. Respond politely and let them know they can "
                "order again anytime.\n\n"
                "Keep it short and friendly — 1-2 sentences."
            )
            self.llm().invoke(prompt)
            return {}

        # Scenario 3: Valid order just priced — ask for confirmation
        if is_valid and has_order_intent:
            pricing_data = state.get("pricing_result", "{}")
            prompt = (
                "You are a Smart Order Assistant. The customer placed an order "
                "and it has been validated and priced. Present the order summary "
                "and ASK the customer to confirm.\n\n"
                "Rules:\n"
                "- Show each item with quantity and price\n"
                "- Show the grand total\n"
                "- Use emojis\n"
                "- Write prices as 'X dollars' not with dollar signs\n"
                "- End by asking: 'Would you like to confirm this order? "
                "Reply yes to confirm or no to cancel.'\n\n"
                f"Pricing data:\n{pricing_data}"
            )
            self.llm().invoke(prompt)
            return {}

        # Scenario 4: Invalid order — show errors
        if has_order_intent and not is_valid:
            validation_data = state.get("validation_result", "{}")
            prompt = (
                "You are a Smart Order Assistant. The customer tried to order "
                "but there were problems. Explain what went wrong and show the menu.\n\n"
                "Rules:\n"
                "- Be friendly and helpful\n"
                "- List the specific errors\n"
                "- Show the menu: Pizza (10 dollars), Burger (8 dollars), Coke (3 dollars)\n"
                "- Maximum 10 of any item\n"
                "- Write prices as 'X dollars' not with dollar signs\n\n"
                f"Validation data:\n{validation_data}"
            )
            self.llm().invoke(prompt)
            return {}

        # Scenario 5: Greeting / general message — respond conversationally
        user_message = _extract_latest_user_message(state)
        prompt = (
            "You are a friendly Smart Order Assistant for a restaurant.\n"
            "The customer sent a message that doesn't seem to be a food order.\n\n"
            "Respond naturally and warmly. Mention our menu:\n"
            "  - Pizza: 10 dollars\n"
            "  - Burger: 8 dollars\n"
            "  - Coke: 3 dollars\n\n"
            "Maximum 10 of any item per order.\n"
            "Keep your response short and friendly — 2-3 sentences max.\n"
            "Do NOT use dollar signs. Write prices as 'X dollars'.\n\n"
            f"Customer message: {user_message}"
        )
        self.llm().invoke(prompt)
        return {}

    # ------------------------------------------------------------------
    # Workflow graph — defines the order of agent execution
    # ------------------------------------------------------------------

    @property
    def workflow(self) -> StateGraph[OrderState]:  # type: ignore[override]
        """Build the LangGraph workflow for the Smart Order Assistant.

        The flow is:
          START → intake → [conditional: what is the intent?]
            ├─ new_order / modification → extraction → validation → [conditional: valid?]
            │                   ├─ valid   → pricing → final_response (ask confirm) → END
            │                   └─ invalid → final_response (show errors) → END
            ├─ confirmation → confirmation → final_response (confirmed) → END
            └─ cancellation / greeting → final_response → END
        """
        graph: StateGraph[OrderState] = StateGraph(OrderState)

        # --- Register all 6 nodes ---
        graph.add_node("intake_node", self._intake_node)
        graph.add_node("extraction_node", self._extraction_node)
        graph.add_node("validation_node", self._validation_node)
        graph.add_node("pricing_node", self._pricing_node)
        graph.add_node("confirmation_node", self._confirmation_node)
        graph.add_node("final_response_node", self._final_response_node)

        # --- Define the edges (connections between nodes) ---

        # Entry point: start with the intake check
        graph.add_edge(START, "intake_node")

        # After intake, use conditional routing:
        graph.add_conditional_edges(
            "intake_node",
            _route_after_intake,
            {
                "extraction_node": "extraction_node",
                "confirmation_node": "confirmation_node",
                "final_response_node": "final_response_node",
            },
        )

        # After extraction, always validate
        graph.add_edge("extraction_node", "validation_node")

        # After validation, use conditional routing:
        graph.add_conditional_edges(
            "validation_node",
            _route_after_validation,
            {
                "pricing_node": "pricing_node",
                "final_response_node": "final_response_node",
            },
        )

        # After pricing, go to final response (which asks for confirmation)
        graph.add_edge("pricing_node", "final_response_node")

        # After confirmation, go to final response (which shows result)
        graph.add_edge("confirmation_node", "final_response_node")

        # Final response is the last node
        graph.add_edge("final_response_node", END)

        return graph

    # ------------------------------------------------------------------
    # Backward-compatible single agent_node (fallback)
    # ------------------------------------------------------------------

    @property
    def agent_node(self) -> Any:
        """Single-agent fallback that can handle the full order flow."""
        menu_str = ", ".join(f"{k} ({v} dollars)" for k, v in MENU.items())
        return create_agent(
            self.llm(),
            tools=self.tools + self.mcp_tools + self._workflow_tools,
            system_prompt=make_system_prompt(
                "You are a Smart Order Assistant for a restaurant.\n"
                "\n"
                "## Menu\n"
                f"{menu_str}\n"
                "\n"
                "## Your Tools\n"
                "\n"
                "1. **Order Extractor** (`extract_order_items`): Extracts food items "
                "and quantities from the customer's message.\n"
                "\n"
                "2. **Order Validator** (`validate_order`): Checks that items are on "
                "the menu and quantities are between 1 and 10.\n"
                "\n"
                "3. **Order Pricer** (`calculate_order_price`): Calculates line-item "
                "prices and the grand total.\n"
                "\n"
                "4. **Order Confirmer** (`confirm_order`): Simulates human-in-the-loop "
                "order confirmation.\n"
                "\n"
                "5. **Response Formatter** (`format_order_response`): Formats the final "
                "response for the customer.\n"
                "\n"
                "## Guidelines\n"
                "\n"
                "- Process orders step by step: extract → validate → price → confirm → respond.\n"
                "- If validation fails, skip pricing and confirmation.\n"
                "- Always be friendly and helpful.\n"
                "- Maximum 10 of any item per order.\n"
            ),
            name="agent",
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _last_ai_content(result: Any) -> str:
    """Extract the last AI message content from a react-agent result dict."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return str(result)


def _extract_latest_user_message(state: OrderState) -> str:
    """Extract the latest user message from the conversation."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _extract_tool_output(result: Any, tool_name: str) -> str:
    """Try to extract the raw output of a specific tool from agent results."""
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if hasattr(msg, "name") and msg.name == tool_name:
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _has_pending_order(state: OrderState) -> bool:
    """Check if there is a pending order in the conversation history.

    Looks at AI messages to see if the assistant previously presented an
    order summary and asked for confirmation.
    """
    messages = state.get("messages", [])
    # Skip the current user message (the latest one) and look at previous AI messages
    skipped_first_human = False
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if not skipped_first_human:
                skipped_first_human = True
                continue
            else:
                break
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_lower = content.lower()
            # Check if the AI was asking for order confirmation
            has_confirm_word = any(
                phrase in content_lower
                for phrase in [
                    "yes to confirm",
                    "reply yes",
                    "confirm this order",
                    "confirm your order",
                    "shall i proceed",
                    "reply yes to confirm",
                ]
            )
            if not has_confirm_word:
                has_confirm_word = "confirm" in content_lower and (
                    "order" in content_lower or "total" in content_lower
                )
            return has_confirm_word
    return False


def _extract_total_from_history(state: OrderState) -> int:
    """Extract the grand total from human-readable AI messages in conversation history.

    Scans AI messages for patterns like "Total: 59 dollars", "Grand Total: 59 dollars".
    Returns the total as an integer, or 0 if not found.
    """
    import re

    messages = state.get("messages", [])
    total_patterns = [
        r"(?:grand\s+)?total[:\s]+(\d+)\s*dollars",
        r"💰\s*(?:grand\s+)?total[:\s]+(\d+)\s*dollars",
        r"total[:\s]+(\d+)",
    ]
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_lower = content.lower()
            for pattern in total_patterns:
                match = re.search(pattern, content_lower)
                if match:
                    try:
                        return int(match.group(1))
                    except (ValueError, IndexError):
                        continue
    return 0


def _extract_order_summary_from_history(state: OrderState) -> str:
    """Extract the order summary text from the AI's confirmation prompt message.

    Looks for the AI message that contains the order summary (the one with
    emojis and line items that asked the user to confirm).
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_lower = content.lower()
            if "confirm" in content_lower and (
                "pizza" in content_lower
                or "burger" in content_lower
                or "coke" in content_lower
                or "total" in content_lower
            ):
                return content
    return ""


def _extract_previous_order_from_history(state: OrderState) -> str:
    """Extract the previous order details from conversation history.

    Looks for the most recent AI message that contains order item details
    (the order summary that was presented to the user). This is used when
    the user requests a modification so we can merge previous items with
    the new changes.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_lower = content.lower()
            # Look for messages that contain order items (menu item names + quantities/prices)
            has_menu_items = any(item in content_lower for item in MENU)
            has_order_indicators = any(
                indicator in content_lower
                for indicator in ["total", "order", "dollars", "quantity", "price"]
            )
            if has_menu_items and has_order_indicators:
                return content
    return "No previous order found."
