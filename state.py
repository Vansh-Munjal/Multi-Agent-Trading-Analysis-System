"""
state.py
--------
Defines the shared AgentState TypedDict that flows through the LangGraph
StateGraph. Every node reads from and writes back to this shared state object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state object that travels through every node in the trading
    analysis graph.

    Fields
    ------
    messages : List[BaseMessage]
        Full conversation / memory buffer (HumanMessage, AIMessage, etc.)
    ticker : str
        The asset symbol under analysis (e.g. "AAPL", "BTC", "SOL").
    context : str
        Raw news / market context text supplied by the user for sentiment
        analysis.
    analyst_reports : Dict[str, Any]
        Keyed store for specialist outputs:
          - "fundamental"  : dict from mock_fundamental_analysis
          - "technical"    : dict from mock_technical_analysis
          - "sentiment"    : dict from mock_sentiment_analysis
          - "risk"         : dict from calculate_risk_score
    final_decision : Optional[str]
        Supervisor's final recommendation string: "Buy", "Hold", or "Sell".
    confidence : Optional[int]
        Supervisor's confidence level (0-100).
    reasoning : Optional[str]
        Detailed reasoning string produced by the supervisor.
    risk_summary : Optional[str]
        Concise risk profile produced by the supervisor.
    next : str
        Routing signal written by the supervisor node; tells the graph which
        node to visit next (one of: "fundamental_analyst", "technical_analyst",
        "sentiment_analyst", "risk_manager", or "END").
    steps_completed : List[str]
        Ordered log of agents that have already run (used to prevent loops
        and drive the progress UI in Streamlit).
    error : Optional[str]
        Populated if any node encounters a non-fatal error so the UI can
        surface it gracefully.
    """

    messages: List[BaseMessage]
    ticker: str
    context: str
    analyst_reports: Dict[str, Any]
    final_decision: Optional[str]
    confidence: Optional[int]
    reasoning: Optional[str]
    risk_summary: Optional[str]
    next: str
    steps_completed: List[str]
    error: Optional[str]
