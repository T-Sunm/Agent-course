from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.core.nodes.caption_node import caption_node
from src.core.memory_manager import session_memory
from src.core.state import ViReAgentState
from src.core.graph_builder.sub_graph import SubGraphBuilder
from src.core.nodes.voting_node import voting_node

class MainGraphBuilder:
    """Builder for the main multi-agent workflow"""
    
    def __init__(self, tools_registry: Dict[str, Any], memory_enabled: bool = True):
        self.tools_registry = tools_registry
        self.memory_enabled = memory_enabled
        self.subgraph_builder = SubGraphBuilder(tools_registry, memory_enabled)
        
    def create_main_workflow(self, 
                       checkpointer = None,
                       thread_id: str = "default"):
        """Create the main multi-agent workflow"""
        
        if checkpointer is None and self.memory_enabled:
            checkpointer = session_memory.get_checkpointer()
        elif checkpointer is None:
            checkpointer = MemorySaver()
        
        main_workflow = StateGraph(ViReAgentState)

        junior_node = self.subgraph_builder.create_junior_subgraph()
        senior_node = self.subgraph_builder.create_senior_subgraph()
        manager_node = self.subgraph_builder.create_manager_subgraph()
    
        # Add nodes
        main_workflow.add_node("caption", caption_node)
        main_workflow.add_node("junior_analyst", junior_node)
        main_workflow.add_node("senior_analyst", senior_node)
        main_workflow.add_node("manager_analyst", manager_node)
        main_workflow.add_node("voting", voting_node)

        main_workflow.add_edge(START, "caption")
        main_workflow.add_edge("caption", "junior_analyst")
        main_workflow.add_edge("caption", "senior_analyst")
        main_workflow.add_edge("caption", "manager_analyst")

        main_workflow.add_edge("junior_analyst", "voting")
        main_workflow.add_edge("senior_analyst", "voting")
        main_workflow.add_edge("manager_analyst", "voting")

        main_workflow.add_edge("voting", END)
        
        return main_workflow.compile(checkpointer=checkpointer)


