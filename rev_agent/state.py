from typing import Annotated, Optional, List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for Agentic RAG workflow"""
   
    messages: Annotated[list, add_messages]
    question: Optional[str]
    conversation_history: Optional[List[Dict[str, str]]]
    
    
    tool_choice: Optional[str]  
    
    
    rag_documents: Optional[list]  
    tavily_results: Optional[str]
    
    
    can_answer_internally: Optional[bool]  
    validation_result: Optional[str]  
    tools_tried: Optional[List[str]]    
