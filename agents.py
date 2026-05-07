"""
agents.py
---------
LangGraph node functions for the Multi-Agent Trading Analysis System.

Each specialist node:
  1. Calls its mock tool to get raw data.
  2. Runs an LCEL chain (SystemMessage + HumanMessage → LLM) to produce a
     natural-language analyst report.
  3. Writes its report into state["analyst_reports"].

The supervisor node:
  Phase A — inspects which reports are done and returns the next routing signal.
  Phase B — once all 4 reports exist, synthesises them via Pydantic structured
             output and writes the final decision into state.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field

from prompts import (
    FUNDAMENTAL_ANALYST_SYSTEM_PROMPT,
    RISK_MANAGER_SYSTEM_PROMPT,
    SENTIMENT_ANALYST_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    TECHNICAL_ANALYST_SYSTEM_PROMPT,
)
from state import AgentState
from tools import (
    calculate_risk_score,
    mock_fundamental_analysis,
    mock_sentiment_analysis,
    mock_technical_analysis,
)

# ─────────────────────────────────────────────────────────────────────────────
# LLM client — shared across all agents
# ─────────────────────────────────────────────────────────────────────────────

def _build_client() -> ChatNVIDIA:
    """Instantiate the ChatNVIDIA client with the project configuration."""
    return ChatNVIDIA(
        model="stepfun-ai/step-3.5-flash",
        api_key=os.environ.get("NVIDIA_API_KEY"),
        temperature=1.0,
        top_p=0.9,
        max_tokens=16384,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model for the supervisor's final structured output
# ─────────────────────────────────────────────────────────────────────────────

class TradingDecision(BaseModel):
    """Structured final trading recommendation produced by the Supervisor."""

    recommendation: str = Field(
        description="Final trading recommendation: one of 'Buy', 'Hold', or 'Sell'."
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence level as an integer from 0 (no conviction) to 100 (maximum conviction).",
    )
    reasoning: str = Field(
        description=(
            "Detailed 3-5 sentence synthesis referencing insights from all four "
            "specialist analysts to justify the recommendation."
        )
    )
    risk_summary: str = Field(
        description="Concise 1-2 sentence summary of the key risk profile for this trade."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper — invoke LLM with a system + human message pair
# ─────────────────────────────────────────────────────────────────────────────

def _llm_call(system_prompt: str, human_message: str, client: ChatNVIDIA) -> str:
    """Run a simple two-message LCEL chain and return the response text."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]
    response = client.invoke(messages)
    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# Helper — parse JSON from supervisor synthesis (handles markdown code fences)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a string that may contain markdown
    code fences (```json ... ```) or plain JSON.
    """
    # Strip markdown fences
    stripped = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    # Find first '{' and last '}'
    start = stripped.find("{")
    end = stripped.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")
    return json.loads(stripped[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Supervisor
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> AgentState:
    """
    Central routing and synthesis node.

    Phase A: Determines which specialist to call next by inspecting which
             reports have already been collected.
    Phase B: When all 4 specialist reports are present, synthesises a final
             structured trading decision.
    """
    client = _build_client()
    reports: dict = state.get("analyst_reports", {})
    ticker: str = state.get("ticker", "UNKNOWN")

    all_specialists = ["fundamental", "technical", "sentiment", "risk"]
    completed = [k for k in all_specialists if k in reports]

    # ── Phase B: all reports ready → synthesise ──────────────────────────────
    if len(completed) == 4:
        reports_text = "\n\n".join(
            f"=== {k.upper()} ANALYST REPORT ===\n{json.dumps(v, indent=2)}"
            for k, v in reports.items()
        )
        human_msg = (
            f"Asset under analysis: **{ticker}**\n\n"
            f"All specialist reports have been collected:\n\n{reports_text}\n\n"
            "Please synthesise these into a final investment recommendation "
            "as a valid JSON object with keys: recommendation, confidence, "
            "reasoning, risk_summary."
        )

        raw_response = _llm_call(SUPERVISOR_SYSTEM_PROMPT, human_msg, client)

        # Attempt structured parse; fall back to regex extraction
        try:
            decision_dict = _extract_json(raw_response)
            decision = TradingDecision(**decision_dict)
        except Exception as parse_err:
            # Graceful degradation — build a conservative Hold decision
            decision = TradingDecision(
                recommendation="Hold",
                confidence=50,
                reasoning=f"Synthesis parsing error: {parse_err}. Raw output: {raw_response[:300]}",
                risk_summary="Unable to fully parse structured output — treat with caution.",
            )

        new_messages = list(state.get("messages", [])) + [
            AIMessage(content=f"Final Decision: {decision.recommendation} ({decision.confidence}% confidence)")
        ]

        return {
            **state,
            "messages": new_messages,
            "next": "END",
            "final_decision": decision.recommendation,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "risk_summary": decision.risk_summary,
            "steps_completed": state.get("steps_completed", []) + ["supervisor_synthesis"],
        }

    # ── Phase A: route to next specialist ────────────────────────────────────
    remaining = [s for s in all_specialists if s not in completed]
    next_specialist_key = remaining[0]  # Follow preferred order

    routing_map = {
        "fundamental": "fundamental_analyst",
        "technical": "technical_analyst",
        "sentiment": "sentiment_analyst",
        "risk": "risk_manager",
    }
    next_node = routing_map[next_specialist_key]

    # Ask LLM to confirm routing (adds explainability; LLM validates choice)
    human_msg = (
        f"Ticker: {ticker}\n"
        f"Reports already collected: {completed or 'None'}\n"
        f"Reports still needed: {remaining}\n"
        "Which specialist should run next? Return ONLY the routing key."
    )
    llm_route = _llm_call(SUPERVISOR_SYSTEM_PROMPT, human_msg, client).strip().lower()

    # Validate LLM routing; fall back to preferred order if invalid
    valid_routes = set(routing_map.values()) | {"synthesize", "end"}
    if llm_route not in valid_routes:
        llm_route = next_node

    new_messages = list(state.get("messages", [])) + [
        AIMessage(content=f"Routing to: {llm_route}")
    ]

    return {
        **state,
        "messages": new_messages,
        "next": llm_route,
        "steps_completed": state.get("steps_completed", []) + [f"supervisor_routing→{llm_route}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Fundamental Analyst
# ─────────────────────────────────────────────────────────────────────────────

def fundamental_analyst_node(state: AgentState) -> AgentState:
    """
    Calls mock_fundamental_analysis for the ticker and generates a
    professional analyst report using the Fundamental Analyst prompt.
    """
    client = _build_client()
    ticker: str = state.get("ticker", "UNKNOWN")

    # Tool call
    raw_data = mock_fundamental_analysis(ticker)

    # LCEL chain
    human_msg = (
        f"Asset: {ticker}\n\n"
        f"Raw fundamental data from our analytics system:\n"
        f"```json\n{json.dumps(raw_data, indent=2)}\n```\n\n"
        "Please write your fundamental analysis report."
    )
    report_text = _llm_call(FUNDAMENTAL_ANALYST_SYSTEM_PROMPT, human_msg, client)

    updated_reports = {**state.get("analyst_reports", {})}
    updated_reports["fundamental"] = {
        "raw_data": raw_data,
        "report": report_text,
    }

    new_messages = list(state.get("messages", [])) + [
        AIMessage(content=f"[Fundamental Analyst] Report complete for {ticker}.")
    ]

    return {
        **state,
        "messages": new_messages,
        "analyst_reports": updated_reports,
        "next": "supervisor",
        "steps_completed": state.get("steps_completed", []) + ["fundamental_analyst"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Technical Analyst
# ─────────────────────────────────────────────────────────────────────────────

def technical_analyst_node(state: AgentState) -> AgentState:
    """
    Calls mock_technical_analysis for the ticker and generates a
    professional technical analysis report using the Technical Analyst prompt.
    """
    client = _build_client()
    ticker: str = state.get("ticker", "UNKNOWN")

    raw_data = mock_technical_analysis(ticker)

    human_msg = (
        f"Asset: {ticker}\n\n"
        f"Raw technical indicator data:\n"
        f"```json\n{json.dumps(raw_data, indent=2)}\n```\n\n"
        "Please write your technical analysis report."
    )
    report_text = _llm_call(TECHNICAL_ANALYST_SYSTEM_PROMPT, human_msg, client)

    updated_reports = {**state.get("analyst_reports", {})}
    updated_reports["technical"] = {
        "raw_data": raw_data,
        "report": report_text,
    }

    new_messages = list(state.get("messages", [])) + [
        AIMessage(content=f"[Technical Analyst] Report complete for {ticker}.")
    ]

    return {
        **state,
        "messages": new_messages,
        "analyst_reports": updated_reports,
        "next": "supervisor",
        "steps_completed": state.get("steps_completed", []) + ["technical_analyst"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Sentiment Analyst
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_analyst_node(state: AgentState) -> AgentState:
    """
    Calls mock_sentiment_analysis on the user-provided context and generates
    a behavioural finance report using the Sentiment Analyst prompt.
    """
    client = _build_client()
    ticker: str = state.get("ticker", "UNKNOWN")
    context: str = state.get("context", "No additional context provided.")

    raw_data = mock_sentiment_analysis(context)

    human_msg = (
        f"Asset: {ticker}\n\n"
        f"User-provided news/context:\n\"{context}\"\n\n"
        f"Raw sentiment analysis results:\n"
        f"```json\n{json.dumps(raw_data, indent=2)}\n```\n\n"
        "Please write your sentiment analysis report."
    )
    report_text = _llm_call(SENTIMENT_ANALYST_SYSTEM_PROMPT, human_msg, client)

    updated_reports = {**state.get("analyst_reports", {})}
    updated_reports["sentiment"] = {
        "raw_data": raw_data,
        "report": report_text,
    }

    new_messages = list(state.get("messages", [])) + [
        AIMessage(content=f"[Sentiment Analyst] Report complete for {ticker}.")
    ]

    return {
        **state,
        "messages": new_messages,
        "analyst_reports": updated_reports,
        "next": "supervisor",
        "steps_completed": state.get("steps_completed", []) + ["sentiment_analyst"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

def risk_manager_node(state: AgentState) -> AgentState:
    """
    Calls calculate_risk_score for the ticker and generates a disciplined
    risk assessment report using the Risk Manager prompt.
    """
    client = _build_client()
    ticker: str = state.get("ticker", "UNKNOWN")

    raw_data = calculate_risk_score(ticker, position_size=100_000)

    human_msg = (
        f"Asset: {ticker}\n\n"
        f"Raw risk metrics from our risk engine:\n"
        f"```json\n{json.dumps(raw_data, indent=2)}\n```\n\n"
        "Please write your risk assessment report."
    )
    report_text = _llm_call(RISK_MANAGER_SYSTEM_PROMPT, human_msg, client)

    updated_reports = {**state.get("analyst_reports", {})}
    updated_reports["risk"] = {
        "raw_data": raw_data,
        "report": report_text,
    }

    new_messages = list(state.get("messages", [])) + [
        AIMessage(content=f"[Risk Manager] Report complete for {ticker}.")
    ]

    return {
        **state,
        "messages": new_messages,
        "analyst_reports": updated_reports,
        "next": "supervisor",
        "steps_completed": state.get("steps_completed", []) + ["risk_manager"],
    }
