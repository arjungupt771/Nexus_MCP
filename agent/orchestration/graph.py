"""Assemble the multi-server orchestration graph.

user_request -> understand_intent --(no tools)--> final_response
                     | needs tools
                plan_tasks -> discover_tools -> select_tools <-+ (repair, bounded)
                                                    | valid plan |
                              schedule_execution -> execute_plan -> aggregate_results
                                                                      | failures, budget left
                                                                prepare_replan --> select_tools
                                                                      | otherwise
                                                                final_response
any LLM failure -> handle_error
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.orchestration.nodes import (
    N_AGGREGATE,
    N_DISCOVER,
    N_ERROR,
    N_EXECUTE,
    N_FINAL,
    N_INTENT,
    N_REPLAN,
    N_SCHEDULE,
    N_SELECT,
    N_TASKS,
    N_USER_REQUEST,
    OrchestrationNodes,
    OrchestrationSettings,
    make_routers,
)
from agent.orchestration.state import OrchestrationState


def build_orchestration_graph(
    nodes: OrchestrationNodes,
    settings: OrchestrationSettings,
) -> CompiledStateGraph:
    after_intent, on_error_else, after_select, after_aggregate = make_routers(settings)
    graph = StateGraph(OrchestrationState)

    for name, fn in (
        (N_USER_REQUEST, nodes.user_request),
        (N_INTENT, nodes.understand_intent),
        (N_TASKS, nodes.plan_tasks),
        (N_DISCOVER, nodes.discover_tools),
        (N_SELECT, nodes.select_tools),
        (N_SCHEDULE, nodes.schedule_execution),
        (N_EXECUTE, nodes.execute_plan),
        (N_AGGREGATE, nodes.aggregate_results),
        (N_REPLAN, nodes.prepare_replan),
        (N_FINAL, nodes.final_response),
        (N_ERROR, nodes.handle_error),
    ):
        graph.add_node(name, fn)

    graph.add_edge(START, N_USER_REQUEST)
    graph.add_edge(N_USER_REQUEST, N_INTENT)
    graph.add_conditional_edges(
        N_INTENT, after_intent, {N_TASKS: N_TASKS, N_FINAL: N_FINAL, N_ERROR: N_ERROR}
    )
    graph.add_conditional_edges(
        N_TASKS, on_error_else(N_DISCOVER), {N_DISCOVER: N_DISCOVER, N_ERROR: N_ERROR}
    )
    graph.add_edge(N_DISCOVER, N_SELECT)
    graph.add_conditional_edges(
        N_SELECT,
        after_select,
        {N_SELECT: N_SELECT, N_SCHEDULE: N_SCHEDULE, N_FINAL: N_FINAL, N_ERROR: N_ERROR},
    )
    graph.add_edge(N_SCHEDULE, N_EXECUTE)
    graph.add_edge(N_EXECUTE, N_AGGREGATE)
    graph.add_conditional_edges(
        N_AGGREGATE, after_aggregate, {N_REPLAN: N_REPLAN, N_FINAL: N_FINAL}
    )
    graph.add_edge(N_REPLAN, N_SELECT)
    graph.add_edge(N_FINAL, END)
    graph.add_edge(N_ERROR, END)

    return graph.compile()
