from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from tavily import TavilyClient
import os

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def get_retriever_tool(retriever):
    """Create retriever tool"""
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_cmo_revenue_playbook",
        "Search and return information about the CMO's Revenue Planning Playbook. "
        "This tool provides insights on revenue planning strategies, marketing metrics "
        "(CAC, LTV, ARR, win rates), channel efficiency optimization, MQL and pipeline calculations, "
        "budget allocation frameworks, team alignment strategies, KPI dashboards, and best practices "
        "for CMOs driving sustainable revenue growth."
    )
    return retriever_tool


@tool
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily for current information, news, and research.
    
    Use this tool when you need:
    - Current events or recent news
    - Real-time information
    - Market research or competitor analysis
    - Verification of facts or statistics
    - Information beyond your knowledge cutoff
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string containing search results with titles, URLs, and content snippets
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )
        
        if not response.get('results'):
            return "No results found for your query."
        
        formatted_results = []
        for idx, result in enumerate(response['results'], 1):
            formatted_results.append(
                f"{idx}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   {result['content']}\n"
            )
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error performing search: {str(e)}"
