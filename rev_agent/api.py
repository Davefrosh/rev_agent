from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from agent import query_agent, stream_agent
import os
import uvicorn
import json

app = FastAPI(
    title="Revenue Planning Agent",
    description="AI assistant for revenue planning and marketing strategy",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question", min_length=1)
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation messages"
    )

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent's generated response")

class HealthResponse(BaseModel):
    status: str
    message: str

class InfoResponse(BaseModel):
    version: str
    model: str
    embedding_model: str
    description: str



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main Agent chat endpoint (Non-streaming)
    
    Processes user query with optional conversation history and returns complete AI response
    """
    try:
       
        history = None
        if request.conversation_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        
        response = query_agent(
            query=request.query,
            conversation_history=history
        )
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming Agent chat endpoint
    
    Streams AI response in real-time as tokens are generated.
    Use Server-Sent Events (SSE) format for frontend consumption.
    
    Example usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/chat/stream?query=...');
    eventSource.onmessage = (event) => {
        console.log(event.data);
    };
    ```
    """
    try:
        history = None
        if request.conversation_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        def generate():
            """Generator function for streaming response"""
            stream_gen = None
            try:
                stream_gen = stream_agent(
                    query=request.query,
                    conversation_history=history
                )
                
                for chunk in stream_gen:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except GeneratorExit:
                
                if stream_gen:
                    try:
                        stream_gen.close()
                    except:
                        pass
                raise
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
            finally:
                
                if stream_gen:
                    try:
                        stream_gen.close()
                    except:
                        pass
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint
    
    Returns service status
    """
    try:
       
        return HealthResponse(
            status="ok",
            message="Agent is running"
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            message=str(e)
        )


@app.get("/info", response_model=InfoResponse)
@app.get("/version", response_model=InfoResponse)
async def info():
    """
    API information endpoint
    """
    return InfoResponse(
        version="1.0.0",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        description="Revenue Planning Agent"
    )



@app.get("/")
async def root():
    """Root endpoint with API documentation link"""
    return {
        "message": "Revenue Planning Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False  
    )

