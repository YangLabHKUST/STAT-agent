"""The staged pipeline as a LangGraph StateGraph.

LangGraph owns control flow, checkpointing, interrupts and resumption; every node is a
thin wrapper around a plain-Python role or guard. Interrupts live only in the ``ask_*``
nodes, which do nothing else, so re-execution on resume is harmless.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stat_agent.agent.nodes.execution import analyze_node, execute_node, generate_node
from stat_agent.agent.nodes.finishing import block_step_node, finish_step_node, finish_turn_node
from stat_agent.agent.nodes.planning import ask_planner_node, plan_node, select_step_node
from stat_agent.agent.nodes.selection import (
    ask_skill_node,
    ask_verifier_node,
    filter_node,
    match_node,
    synthesize_card_node,
    verify_node,
)
from stat_agent.agent.state import Context, TurnState

NODES = {
    "plan": (plan_node, ("ask_planner", "select_step")),
    "ask_planner": (ask_planner_node, ("plan", "finish_turn")),
    "select_step": (select_step_node, ("filter", "finish_step", "finish_turn")),
    "filter": (filter_node, ("match",)),
    "match": (match_node, ("verify", "ask_skill", "synthesize_card", "finish_step", "finish_turn")),
    "ask_skill": (ask_skill_node, ("verify", "ask_skill", "block_step")),
    "synthesize_card": (synthesize_card_node, ("generate", "verify", "block_step", "finish_step", "finish_turn")),
    "verify": (verify_node, ("generate", "ask_verifier", "block_step", "finish_step", "finish_turn")),
    "ask_verifier": (ask_verifier_node, ("verify", "block_step")),
    "generate": (generate_node, ("execute", "finish_step", "analyze", "finish_turn")),
    "execute": (execute_node, ("analyze", "generate", "finish_step", "finish_turn")),
    "analyze": (analyze_node, ("finish_step", "finish_turn")),
    "block_step": (block_step_node, ("finish_step",)),
    "finish_step": (finish_step_node, ("select_step", "finish_turn")),
    "finish_turn": (finish_turn_node, ()),
}

INTERRUPT_NODES = ("ask_planner", "ask_skill", "ask_verifier")


def build_graph(checkpointer=None):
    builder = StateGraph(TurnState, context_schema=Context)
    for name, (fn, dests) in NODES.items():
        builder.add_node(name, fn, destinations=dests or None)
    builder.add_edge(START, "plan")
    builder.add_edge("finish_turn", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = ["build_graph", "NODES", "INTERRUPT_NODES"]
