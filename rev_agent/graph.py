from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from state import AgentState
from nodes import create_nodes
from embeddings_setup import get_retriever
from tools_setup import tavily_search


def create_graph():
    """Create and compile the Agentic RAG workflow graph with validation loop"""
    
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    
    retriever = get_retriever()
    
    
    nodes = create_nodes(llm, retriever, tavily_search)
    
    
    workflow = StateGraph(AgentState)
    
    
    workflow.add_node("assess_internal_knowledge", nodes["assess_internal_knowledge"])
    workflow.add_node("route_query", nodes["route_query"])
    workflow.add_node("execute_rag_tool", nodes["execute_rag_tool"])
    workflow.add_node("execute_tavily_tool", nodes["execute_tavily_tool"])
    workflow.add_node("execute_both_tools", nodes["execute_both_tools"])
    workflow.add_node("validate_and_reason", nodes["validate_and_reason"])
    workflow.add_node("generate_response", nodes["generate_response"])
    
    
    
    
    workflow.add_edge(START, "assess_internal_knowledge")
    
    
    workflow.add_conditional_edges(
        "assess_internal_knowledge",
        nodes["internal_knowledge_decision"],
        {
            "generate_directly": "generate_response",
            "route_to_tools": "route_query"
        }
    )
    
    
    workflow.add_conditional_edges(
        "route_query",
        nodes["route_decision"],
        {
            "use_rag": "execute_rag_tool",
            "use_tavily": "execute_tavily_tool",
            "use_both": "execute_both_tools",
            "use_none": "generate_response"
        }
    )
    
    
    workflow.add_edge("execute_rag_tool", "validate_and_reason")
    workflow.add_edge("execute_tavily_tool", "validate_and_reason")
    workflow.add_edge("execute_both_tools", "validate_and_reason")
    
    
    workflow.add_conditional_edges(
        "validate_and_reason",
        nodes["validation_decision"],
        {
            "generate": "generate_response",
            "try_rag": "execute_rag_tool",
            "try_tavily": "execute_tavily_tool", 
            "generate_llm": "generate_response" 
        }
    )
    
    
    workflow.add_edge("generate_response", END)
    
    
    app = workflow.compile()
    
    return app