"""Assemble the LangGraph agent workflow.

user_input -> prepare_context -> invoke_llm -> detect_tool_calls
    detect_tool_calls --(tool calls, budget left)--> execute_tools
                          -> process_results -> prepare_context  (loop)
    detect_tool_calls --(no tool calls)-----------> final_response
    detect_tool_calls --(budget exhausted)--------> limit_reached
    detect_tool_calls --(provider error)----------> handle_error
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.graph.nodes import (
    NODE_DETECT,
    NODE_ERROR,
    NODE_EXECUTE,
    NODE_FINAL,
    NODE_INVOKE,
    NODE_LIMIT,
    NODE_PREPARE,
    NODE_PROCESS,
    NODE_USER_INPUT,
    AgentNodes,
    make_router,
)
from agent.graph.state import AgentState


def build_agent_graph(nodes: AgentNodes, *, max_iterations: int) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node(NODE_USER_INPUT, nodes.user_input)
    graph.add_node(NODE_PREPARE, nodes.prepare_context)
    graph.add_node(NODE_INVOKE, nodes.invoke_llm)
    graph.add_node(NODE_DETECT, nodes.detect_tool_calls)
    graph.add_node(NODE_EXECUTE, nodes.execute_tools)
    graph.add_node(NODE_PROCESS, nodes.process_results)
    graph.add_node(NODE_FINAL, nodes.final_response)
    graph.add_node(NODE_LIMIT, nodes.limit_reached)
    graph.add_node(NODE_ERROR, nodes.handle_error)

    graph.add_edge(START, NODE_USER_INPUT)
    graph.add_edge(NODE_USER_INPUT, NODE_PREPARE)
    graph.add_edge(NODE_PREPARE, NODE_INVOKE)
    graph.add_edge(NODE_INVOKE, NODE_DETECT)
    graph.add_conditional_edges(
        NODE_DETECT,
        make_router(max_iterations),
        {
            NODE_EXECUTE: NODE_EXECUTE,
            NODE_FINAL: NODE_FINAL,
            NODE_LIMIT: NODE_LIMIT,
            NODE_ERROR: NODE_ERROR,
        },
    )
    graph.add_edge(NODE_EXECUTE, NODE_PROCESS)
    graph.add_edge(NODE_PROCESS, NODE_PREPARE)
    graph.add_edge(NODE_FINAL, END)
    graph.add_edge(NODE_LIMIT, END)
    graph.add_edge(NODE_ERROR, END)

    return graph.compile()
