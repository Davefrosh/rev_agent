from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
from state import AgentState


def create_nodes(llm, retriever, tavily_search_tool):
    """Create all node functions for Agentic RAG workflow"""
    
    
    class InternalKnowledgeAssessment(BaseModel):
        """Assessment of whether AI Brain can answer with internal knowledge"""
        can_answer: bool = Field(
            description="Whether the AI can confidently answer this query using internal knowledge"
        )
        confidence: Literal["high", "medium", "low"] = Field(
            description="Confidence level in the internal answer"
        )
        reasoning: str = Field(
            description="Brief explanation of the assessment"
        )
    
    structured_assessor = llm.with_structured_output(InternalKnowledgeAssessment)
    
    assessment_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI Brain assessing whether you can answer a query using your internal knowledge.

Consider whether you have sufficient knowledge to provide a complete, accurate answer without external tools.

**Can answer with internal knowledge:**
- General concepts, definitions, frameworks, methodologies
- Well-established business practices and strategies
- Common marketing metrics and their calculations
- General advice and explanations

**Need external tools:**
- Specific current market data or recent statistics
- Real-time information or latest trends
- Specific company examples or case studies from the knowledge base
- Questions requiring the CMO Revenue Planning Playbook content

Be honest about your limitations. It's better to use tools than provide incomplete answers."""),
        ("human", "Query: {question}")
    ])
    
    assessment_chain = assessment_prompt | structured_assessor
    
    def assess_internal_knowledge(state: AgentState) -> dict:
        """AI Brain first attempts to assess if it can answer with internal knowledge"""
        messages = state["messages"]
        question = messages[-1].content
        
        
        assessment = assessment_chain.invoke({"question": question})
        
        
        return {
            "question": question,
            "can_answer_internally": assessment.can_answer,
            "tools_tried": []
        }
    
    def internal_knowledge_decision(state: AgentState) -> str:
        """Decide whether to generate response or route to tools"""
        can_answer = state.get("can_answer_internally", False)
        
        if can_answer:
            return "generate_directly"  
        else:
            return "route_to_tools"  
    
    
    class RouteDecision(BaseModel):
        """Router decision for tool selection"""
        tool_choice: Literal["rag", "tavily", "both", "none"] = Field(
            description="Which tool(s) to use: 'rag' for knowledge base, 'tavily' for web search, 'both' for comprehensive answers, 'none' for direct LLM response"
        )
        reasoning: str = Field(
            description="Brief explanation of why this routing decision was made"
        )
    
    structured_router = llm.with_structured_output(RouteDecision)
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent routing system for a Revenue Planning AI Assistant.

Your job is to analyze the user's query and decide which tool(s) to use:

**RAG Tool (Knowledge Base):**
- Use for: Revenue planning strategies, marketing frameworks, metrics (CAC, LTV, ARR, MQL), 
  budget allocation, channel optimization, team alignment, KPI dashboards
- Contains: CMO Revenue Planning Playbook with proven strategies and best practices

**Tavily Tool (Web Search):**
- Use for: Current events, recent news, real-time market data, latest trends, 
  competitor analysis, recent statistics, time-sensitive information

**Both Tools:**
- Use when: Query requires both foundational knowledge AND current market context
- Example: "What's our revenue strategy for AI startups given recent market conditions?"
- Use when one tool alone might not provide sufficient comprehensive answer

**No Tools (Direct LLM):**
- Use for: General questions, greetings, clarifications, simple explanations
- When: Query doesn't require specific knowledge base or current information

Analyze the query and make the best routing decision."""),
        ("human", "User query: {question}")
    ])
    
    router_chain = router_prompt | structured_router
    
    def route_query(state: AgentState) -> dict:
        """Analyze query and decide which tool(s) to use"""
        messages = state["messages"]
        question = messages[-1].content
        
        
        decision = router_chain.invoke({"question": question})
        
        return {
            "question": question,
            "tool_choice": decision.tool_choice
        }
    
    
    
    def execute_rag_tool(state: AgentState) -> dict:
        """Execute RAG retrieval from knowledge base"""
        question = state["question"]
        tools_tried = state.get("tools_tried", [])
        
        try:
            documents = retriever.invoke(question)
            
            if "rag" not in tools_tried:
                tools_tried.append("rag")
            
            return {
                "rag_documents": documents if documents else [],
                "tools_tried": tools_tried
            }
        except Exception as e:
            print(f"RAG retrieval error: {str(e)}")
            
            if "rag" not in tools_tried:
                tools_tried.append("rag")
            
            return {
                "rag_documents": [],
                "tools_tried": tools_tried
            }
    
    def execute_tavily_tool(state: AgentState) -> dict:
        """Execute Tavily web search"""
        question = state["question"]
        tools_tried = state.get("tools_tried", [])
        
        try:
            results = tavily_search_tool.invoke(question)
            
            if "tavily" not in tools_tried:
                tools_tried.append("tavily")
            
            return {
                "tavily_results": results if results else "",
                "tools_tried": tools_tried
            }
        except Exception as e:
            print(f"Tavily search error: {str(e)}")
           
            if "tavily" not in tools_tried:
                tools_tried.append("tavily")
            
            return {
                "tavily_results": "",
                "tools_tried": tools_tried
            }
    
    def execute_both_tools(state: AgentState) -> dict:
        """Execute both RAG and Tavily tools in parallel"""
        question = state["question"]
        tools_tried = state.get("tools_tried", [])
        
        
        rag_docs = []
        try:
            rag_docs = retriever.invoke(question)
        except Exception as e:
            print(f"RAG retrieval error: {str(e)}")
        
        
        tavily_res = ""
        try:
            tavily_res = tavily_search_tool.invoke(question)
        except Exception as e:
            print(f"Tavily search error: {str(e)}")
        
        
        if "rag" not in tools_tried:
            tools_tried.append("rag")
        if "tavily" not in tools_tried:
            tools_tried.append("tavily")
        
        return {
            "rag_documents": rag_docs if rag_docs else [],
            "tavily_results": tavily_res if tavily_res else "",
            "tools_tried": tools_tried
        }
    
    
    
    class ValidationResult(BaseModel):
        """Validation assessment of tool outputs"""
        is_sufficient: bool = Field(
            description="Whether the available tool outputs are sufficient to answer the query"
        )
        next_action: Literal["generate", "try_rag", "try_tavily", "generate_llm"] = Field(
            description="Next action: 'generate' if sufficient, 'try_rag' to retry with RAG, 'try_tavily' to retry with Tavily, 'generate_llm' to use LLM knowledge"
        )
        reasoning: str = Field(
            description="Brief explanation of the validation decision"
        )
    
    structured_validator = llm.with_structured_output(ValidationResult)
    
    validator_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality validator for a Revenue Planning AI Assistant.

Your job is to assess whether the tool outputs are sufficient to answer the user's query.

**Evaluation Criteria:**
1. **Relevance**: Do the results directly address the query?
2. **Completeness**: Is there enough information to provide a comprehensive answer?
3. **Quality**: Are the results meaningful and not just error messages or empty responses?

**Decision Rules:**
- If you have good results from both RAG and Tavily → next_action: "generate"
- If you have good results from one tool → next_action: "generate"
- If RAG is empty/poor but query needs knowledge base → next_action: "try_rag"
- If Tavily is empty/poor but query needs current info → next_action: "try_tavily"
- If both tools failed or returned poor results → next_action: "generate_llm" (use LLM's own knowledge)

Be pragmatic: Don't demand perfection. If the results are reasonably useful, proceed to generate."""),
        ("human", """User query: {question}

RAG Documents Available: {has_rag}
RAG Content Sample: {rag_sample}

Tavily Results Available: {has_tavily}
Tavily Content Sample: {tavily_sample}

Tool Choice Was: {tool_choice}

Assess the quality and decide the next action.""")
    ])
    
    validator_chain = validator_prompt | structured_validator
    
    def validate_and_reason(state: AgentState) -> dict:
        """Validate tool outputs and decide next action"""
        question = state["question"]
        tool_choice = state.get("tool_choice", "none")
        rag_docs = state.get("rag_documents", [])
        tavily_res = state.get("tavily_results", "")
        tools_tried = state.get("tools_tried", [])
        
        
        has_rag = rag_docs is not None and len(rag_docs) > 0
        has_tavily = tavily_res is not None and isinstance(tavily_res, str) and len(tavily_res.strip()) > 0
        
        
        rag_sample = ""
        if has_rag:
            first_doc = rag_docs[0]
            content = first_doc.page_content if hasattr(first_doc, 'page_content') else str(first_doc)
            rag_sample = content[:200] + "..." if len(content) > 200 else content
        
        tavily_sample = ""
        if has_tavily:
            tavily_sample = tavily_res[:200] + "..." if len(tavily_res) > 200 else tavily_res
        
        
        validation = validator_chain.invoke({
            "question": question,
            "has_rag": "Yes" if has_rag else "No",
            "rag_sample": rag_sample or "None",
            "has_tavily": "Yes" if has_tavily else "No",
            "tavily_sample": tavily_sample or "None",
            "tool_choice": tool_choice
        })
        
       
        validation_result = "sufficient" if validation.is_sufficient else "insufficient"
        
        return {
            "validation_result": validation_result
        }
    
    
    
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized Revenue Planning AI Assistant for CMOs and marketing leaders.

**Your Expertise:**
- Revenue planning strategies and frameworks
- Marketing metrics (CAC, LTV, ARR, MQL, pipeline calculations)
- Channel efficiency optimization and budget allocation
- Team alignment and KPI dashboards
- Current market trends and competitive intelligence

**Your Tools:**
1. **CMO Revenue Planning Playbook** - Your primary knowledge base with proven strategies
2. **Real-time web search** - For current events, trends, and latest market data

**Response Guidelines:**
- Be concise but comprehensive (aim for 3-5 sentences for simple queries, more for complex ones)
- Use data-driven insights and specific examples when available
- When using web search results, cite sources naturally (e.g., "According to recent data...")
- When synthesizing from multiple sources, integrate them smoothly
- If you don't have enough information, acknowledge it rather than guessing
- Maintain a professional, advisory tone suitable for C-level executives

{context_instruction}

{history_context}"""),
        ("human", "{question}")
    ])
    
    generator_chain = generator_prompt | llm | StrOutputParser()
    
    def generate_response(state: AgentState) -> dict:
        """Generate final response using available context"""
        question = state["question"]
        rag_docs = state.get("rag_documents", [])
        tavily_res = state.get("tavily_results", "")
        conversation_history = state.get("conversation_history", [])
        tool_choice = state.get("tool_choice", "none")
        
        
        context_parts = []
        
        if rag_docs is not None and len(rag_docs) > 0:
            rag_context = "\n\n".join([
                doc.page_content if hasattr(doc, 'page_content') else str(doc)
                for doc in rag_docs[:5]  
            ])
            context_parts.append(f"**Knowledge Base Context:**\n{rag_context}")
        
        if tavily_res is not None and isinstance(tavily_res, str) and len(tavily_res.strip()) > 0:
            context_parts.append(f"**Current Web Search Results:**\n{tavily_res}")
        
        if context_parts:
            context_instruction = "Use the following context to inform your response:\n\n" + "\n\n---\n\n".join(context_parts)
        else:
            context_instruction = "Use your expertise and training to provide a helpful response based on your knowledge of revenue planning and marketing strategy."
        
       
        history_text = ""
        if conversation_history:
            history_text = "\n**Previous Conversation:**\n"
            for msg in conversation_history[-6:]:  
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_text += f"{role.capitalize()}: {content}\n"
            history_text += "\nConsider this conversation history for contextually relevant answers."
        
        
        generation = generator_chain.invoke({
            "question": question,
            "context_instruction": context_instruction,
            "history_context": history_text
        })
        
        return {
            "messages": [generation]
        }
    
    
    
    def route_decision(state: AgentState) -> str:
        """Determine which tool execution node to route to"""
        tool_choice = state.get("tool_choice", "none")
        
        if tool_choice == "rag":
            return "use_rag"
        elif tool_choice == "tavily":
            return "use_tavily"
        elif tool_choice == "both":
            return "use_both"
        else:  
            return "use_none"
    
    def validation_decision(state: AgentState) -> str:
        """Determine next action based on validation - with retry logic"""
        question = state["question"]
        tool_choice = state.get("tool_choice", "none")
        rag_docs = state.get("rag_documents", [])
        tavily_res = state.get("tavily_results", "")
        tools_tried = state.get("tools_tried", [])
        validation_result = state.get("validation_result", None)
        
        has_rag = rag_docs is not None and len(rag_docs) > 0
        has_tavily = tavily_res is not None and isinstance(tavily_res, str) and len(tavily_res.strip()) > 0
        
        
        if validation_result == "sufficient":
            return "generate"
        
        
        if "rag" in tools_tried and "tavily" in tools_tried:
            return "generate_llm"  
        
        
        if validation_result == "insufficient":
            
            if "rag" in tools_tried and "tavily" not in tools_tried:
                return "try_tavily"
            
            elif "tavily" in tools_tried and "rag" not in tools_tried:
                return "try_rag"
            
            elif not tools_tried:
                if tool_choice == "rag":
                    return "try_tavily"
                else:
                    return "try_rag"
        
        
        return "generate"  
    
    
    
    return {
        
        "assess_internal_knowledge": assess_internal_knowledge,
        "route_query": route_query,
        "execute_rag_tool": execute_rag_tool,
        "execute_tavily_tool": execute_tavily_tool,
        "execute_both_tools": execute_both_tools,
        "validate_and_reason": validate_and_reason,
        "generate_response": generate_response,
        
        
        "internal_knowledge_decision": internal_knowledge_decision,
        "route_decision": route_decision,
        "validation_decision": validation_decision
    }
