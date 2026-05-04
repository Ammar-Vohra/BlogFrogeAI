from src.nodes.orchestrator import orchestrator_node, fanout
from src.nodes.worker import worker_node
from src.nodes.router import router_node, route_next
from src.nodes.research import research_node
from langgraph.graph import StateGraph, START, END
from src.models.schemas import State
from src.graphs.reducer_graph import reducer_subgraph



def MainGraph():
    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker", worker_node)
    g.add_node("reducer", reducer_subgraph())

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
    g.add_edge("research", "orchestrator")

    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)

    return g.compile()