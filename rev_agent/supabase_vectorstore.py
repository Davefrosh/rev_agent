from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import uuid

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    RETRIEVER_K
)


class SupabaseVectorStore:
    """Vector store using Supabase pgvector for document embeddings"""
    
    def __init__(self):
        """Initialize Supabase client and OpenAI embeddings"""
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.table_name = "rag_table"
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to Supabase with embeddings
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        for i, doc in enumerate(documents):
            try:
                
                embedding = self.embeddings.embed_query(doc.page_content)
                
                
                doc_data = {
                    "id": str(uuid.uuid4()),
                    "content": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_index": i
                }
                
                
                result = self.supabase.table(self.table_name).insert(doc_data).execute()
                doc_ids.append(doc_data["id"])
                
            except Exception as e:
                continue
        
        return doc_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = RETRIEVER_K,
        threshold: float = 0.2
    ) -> List[Document]:
        """
        Perform similarity search using Supabase match_documents function
        
        Args:
            query: Search query text
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of matching Document objects
        """
        
        query_embedding = self.embeddings.embed_query(query)
        
        
        try:
            result = self.supabase.rpc(
                "match_rag_table",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": threshold,
                    "match_count": k
                }
            ).execute()
            
            
            documents = []
            for row in result.data:
                doc = Document(
                    page_content=row["content"],
                    metadata=row.get("metadata", {})
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in Supabase"""
        try:
            result = self.supabase.table(self.table_name).select("id", count="exact").execute()
            return result.count
        except Exception as e:
            return 0
    
    def clear_documents(self):
        """Delete all documents from Supabase (use with caution!)"""
        try:
            result = self.supabase.table(self.table_name).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        except Exception as e:
            pass


def get_supabase_retriever():
    """Get a retriever that uses Supabase for similarity search"""
    
    class SupabaseRetriever:
        """Custom retriever wrapper for Supabase"""
        
        def __init__(self):
            self.vectorstore = SupabaseVectorStore()
        
        def get_relevant_documents(self, query: str) -> List[Document]:
            """Get relevant documents for a query"""
            return self.vectorstore.similarity_search(query, k=RETRIEVER_K)
        
        def invoke(self, query: str) -> List[Document]:
            """Invoke method for LangChain compatibility"""
            return self.get_relevant_documents(query)
    
    return SupabaseRetriever()
