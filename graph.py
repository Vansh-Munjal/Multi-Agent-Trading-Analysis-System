"""
graph.py
--------
Defines and compiles the LangGraph StateGraph for the Multi-Agent Trading
Analysis System.

Architecture
------------
                    ┌─────────────────────────────────────┐
                    │           Supervisor Agent           │
                    │  (routes to specialist or END)       │
                    └──────────┬───────────────────────────┘
                               │ conditional edges
            ┌──────────────────┼─────────────────────────────────┐
            ▼                  ▼               ▼                  ▼
  Fundamental Analyst   Technical Analyst  Sentiment Analyst  Risk Manager
            │                  │               │                  │
            └──────────────────┴───────────────┴──────────────────┘
                                      │
                              (all return to supervisor)
                                      │
                                    END

Flow
----
1. START → supervisor  (initial routing decision)
2. supervisor routes to one of the 4 specialist nodes.
3. Specialist runs, writes report, sets next="supervisor".
4. Control returns to supervisor.
5. After all 4 reports are written, supervisor synthesises and sets next="END".
6. Graph terminates.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents import (
    fundamental_analyst_node,
    risk_manager_node,
    sentiment_analyst_node,
    supervisor_node,
    technical_analyst_node,
)
from state import AgentState

# ─────────────────────────────────────────────────────────────────────────────
# Routing function — reads state["next"] set by supervisor
# ─────────────────────────────────────────────────────────────────────────────

def route_from_supervisor(state: AgentState) -> str:
    """
    Conditional edge function called after every supervisor execution.

    Returns the name of the next node to execute based on state["next"].
    Valid values written by supervisor_node:
      - "fundamental_analyst"
      - "technical_analyst"
      - "sentiment_analyst"
      - "risk_manager"
      - "END"
    """
    next_node = state.get("next", "END")

    # Normalise common variants
    normalised = next_node.lower().strip()
    if normalised in ("end", "synthesize", "synthesise", "finish"):
        return END

    valid_nodes = {
        "fundamental_analyst",
        "technical_analyst",
        "sentiment_analyst",
        "risk_manager",
    }
    if normalised in valid_nodes:
        return normalised

    # Default safety: go back to supervisor for re-routing
    return "supervisor"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the full trading analysis StateGraph.

    Returns
    -------
    StateGraph
        A compiled LangGraph application ready to be invoked with an
        initial AgentState dictionary.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("fundamental_analyst", fundamental_analyst_node)
    graph.add_node("technical_analyst", technical_analyst_node)
    graph.add_node("sentiment_analyst", sentiment_analyst_node)
    graph.add_node("risk_manager", risk_manager_node)

    # ── Entry edge: START → supervisor ────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Conditional edges from supervisor based on state["next"] ──────────────
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "fundamental_analyst": "fundamental_analyst",
            "technical_analyst": "technical_analyst",
            "sentiment_analyst": "sentiment_analyst",
            "risk_manager": "risk_manager",
            END: END,
        },
    )

    # ── All specialist nodes return to supervisor ─────────────────────────────
    for specialist in [
        "fundamental_analyst",
        "technical_analyst",
        "sentiment_analyst",
        "risk_manager",
    ]:
        graph.add_edge(specialist, "supervisor")

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled graph (imported by main.py)
# ─────────────────────────────────────────────────────────────────────────────

trading_graph = build_graph()
