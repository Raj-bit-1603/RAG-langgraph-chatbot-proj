from langgraph.graph import StateGraph, END
from workflow.nodes import *

def build_graph():

    graph = StateGraph(dict)

    graph.add_node("input", input_processing)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("validate", context_validation)
    graph.add_node("generate", generation_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("format", formatter_node)

    graph.set_entry_point("input")

    graph.add_edge("input", "retrieve")
    graph.add_edge("retrieve", "validate")

    graph.add_conditional_edges(
        "validate",
        lambda x: x["valid"],
        {
            True: "generate",
            False: "fallback"
        }
    )

    graph.add_edge("generate", "format")
    graph.add_edge("fallback", "format")
    graph.add_edge("format", END)

    return graph.compile()