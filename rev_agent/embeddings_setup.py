from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from supabase_vectorstore import get_supabase_retriever, SupabaseVectorStore
import os
from config import RETRIEVER_K


def load_and_create_vectorstore():
    """
    Load vectorstore from Supabase
    NOTE: Documents must be migrated first using migrate_to_supabase.py
    """
    try:
        vectorstore = SupabaseVectorStore()
        doc_count = vectorstore.get_document_count()
        
        if doc_count == 0:
            raise ValueError("No documents found in Supabase. Run migration first: python migrate_to_supabase.py")
        
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Error loading Supabase vectorstore: {str(e)}")


def get_retriever():
    """Get retriever from Supabase vectorstore"""
    retriever = get_supabase_retriever()
    return retriever
