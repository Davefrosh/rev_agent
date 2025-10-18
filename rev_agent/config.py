import os
from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")


SUPABASE_URL = os.getenv("SUPABASE_URL", "https://keblvjnepumswxlfgquv.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_SERVICE_KEY not found in .env file")


LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "revenue-planning-agent")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    print(f"[OK] LangSmith tracing enabled - Project: {LANGCHAIN_PROJECT}")
else:
    print("[WARNING] LANGCHAIN_API_KEY not set - LangSmith tracing disabled")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


DOCX_PATH = "The CMO's Revenue Planning Playbook_ From Strategy to Sustainable Growth.docx"


EMBEDDING_MODEL = "text-embedding-3-small"  
EMBEDDING_DIMENSIONS = 1536


CHUNK_SIZE = 600 
CHUNK_OVERLAP = 200 
RETRIEVER_K = 5  
