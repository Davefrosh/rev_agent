from typing import List, Dict, Generator
from langchain_core.messages import HumanMessage
from graph import create_graph

_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_graph()
    return _agent_graph


def query_agent(query: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """
    Query the Agentic RAG agent with optional conversation history
    
    The agent dynamically routes queries to appropriate tools:
    - RAG (knowledge base) for revenue planning strategies
    - Tavily (web search) for current market information
    - Both tools for comprehensive answers
    - Direct LLM response for general queries
    """
    
    agent = get_agent()
    

    inputs = {
        "messages": [HumanMessage(content=query)],
        "conversation_history": conversation_history or [],
        "question": None,
        "tool_choice": None,
        "rag_documents": None,
        "tavily_results": None,
        "can_answer_internally": None,
        "validation_result": None,
        "tools_tried": None
    }
    
    try:
        result = None
        
        for output in agent.stream(inputs):
            for key, value in output.items():
                
                if key == "generate_response":
                    result = value["messages"][-1]
        
        if result:
            return result
        else:
            return "Sorry, I couldn't generate a response. Please try again."
            
    except Exception as e:
        return f"Error processing query: {str(e)}"


def stream_agent(query: str, conversation_history: List[Dict[str, str]] = None) -> Generator[str, None, None]:
    """
    Stream the Agentic RAG agent response token by token
    
    Yields individual tokens as they are generated, providing real-time feedback
    
    Args:
        query: User's question
        conversation_history: Optional conversation context
        
    Yields:
        str: Individual response tokens or status updates
    """
    
    agent = get_agent()
    
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "conversation_history": conversation_history or [],
        "question": None,
        "tool_choice": None,
        "rag_documents": None,
        "tavily_results": None,
        "can_answer_internally": None,
        "validation_result": None,
        "tools_tried": None
    }
    
    try:
        response_generated = False
        
        for output in agent.stream(inputs):
            for node_name, value in output.items():
                
                
                if node_name == "route_query":
                    tool_choice = value.get("tool_choice", "unknown")
                    yield f"[ROUTING: {tool_choice}]\n"
                
                elif node_name in ["execute_rag_tool", "execute_tavily_tool", "execute_both_tools"]:
                    yield f"[RETRIEVING...]\n"
                
                
                elif node_name == "generate_response":
                    response = value["messages"][-1]
                    if response:
                        response_generated = True
                        
                        if isinstance(response, str):
                            yield response
                        elif hasattr(response, 'content'):
                            yield response.content
                        else:
                            yield str(response)
                        
                        break  
            
            
            if response_generated:
                break
        
        
        if not response_generated:
            yield "Sorry, I couldn't generate a response. Please try again."
        
    except Exception as e:
        yield f"Error: {str(e)}"

