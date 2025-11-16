"""
Utility functions for the RAG-based QnA agent.

Includes API calls, FAISS indexing, and semantic search functions.
"""

import os
import requests
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

logger = logging.getLogger(__name__)

# API Configuration
MESSAGES_API_URL = os.getenv(
    "MESSAGES_API_URL",
    "https://november7-730026606190.europe-west1.run.app/messages/"
)
MESSAGES_API_KEY = os.getenv("MESSAGES_API_KEY")


def fetch_all_messages(api_url: str = MESSAGES_API_URL, limit: int = 10000) -> List[Dict]:
    """
    Fetch all messages from the API.
    
    Args:
        api_url: URL of the messages API endpoint
        limit: Maximum number of messages to fetch
        
    Returns:
        List of message dictionaries
        
    Raises:
        Exception: If API request fails
    """
    try:
        logger.info(f"Fetching messages from {api_url}")
        
        # Prepare headers with authentication if API key is available
        headers = {}
        if MESSAGES_API_KEY:
            headers["Authorization"] = f"Bearer {MESSAGES_API_KEY}"
        
        response = requests.get(
            api_url,
            params={"skip": 0, "limit": limit},
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        messages = data.get("items", [])
        logger.info(f"Fetched {len(messages)} messages from API")
        return messages
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise


def messages_to_documents(messages: List[Dict], strategy: str = "individual") -> List[Document]:
    """
    Convert API messages to LangChain Documents for FAISS indexing.
    
    Strategies:
    - "individual": Each message becomes a separate document
    - "aggregated": Group all messages by user with chunking and overlap
    - "hybrid": Create both individual and aggregated documents
    
    Args:
        messages: List of raw message dictionaries from API
        strategy: Document creation strategy ("individual", "aggregated", or "hybrid")
        
    Returns:
        List of Document objects
    """
    if strategy == "individual":
        return _create_individual_documents(messages)
    elif strategy == "aggregated":
        return _create_aggregated_documents(messages)
    elif strategy == "hybrid":
        individual = _create_individual_documents(messages)
        aggregated = _create_aggregated_documents(messages)
        logger.info(f"Created {len(individual)} individual + {len(aggregated)} aggregated = {len(individual) + len(aggregated)} total documents")
        return individual + aggregated
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _create_individual_documents(messages: List[Dict]) -> List[Document]:
    """Create one document per message."""
    documents = []
    
    for msg in messages:
        # Create document content with member context
        user_name = msg.get("user_name", "Unknown")
        message_text = msg.get("message", "")
        timestamp = msg.get("timestamp", "")
        
        # Enriched content for better semantic search
        content = f"{user_name} says: {message_text}"
        
        # Add metadata for filtering and citation
        metadata = {
            "user_name": user_name,
            "user_id": msg.get("user_id", ""),
            "timestamp": timestamp,
            "message": message_text,
            "doc_type": "individual",
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    logger.info(f"Converted {len(documents)} messages to individual documents")
    return documents


def _create_aggregated_documents(messages: List[Dict]) -> List[Document]:
    """
    Create aggregated documents per user with chunking and overlap.
    
    This helps answer questions that require information from multiple messages,
    like "How many cars does X have?" by ensuring all of a user's messages
    can be retrieved together.
    """
    from collections import defaultdict
    
    # Group messages by user
    user_messages = defaultdict(list)
    for msg in messages:
        user_name = msg.get("user_name", "Unknown")
        user_messages[user_name].append(msg)
    
    documents = []
    chunk_size = 3  # Number of messages per chunk
    overlap = 1     # Number of overlapping messages between chunks
    
    for user_name, msgs in user_messages.items():
        if not msgs:
            continue
        
        # Sort by timestamp for chronological order
        msgs_sorted = sorted(msgs, key=lambda x: x.get("timestamp", ""))
        
        # Create overlapping chunks
        for i in range(0, len(msgs_sorted), chunk_size - overlap):
            chunk_msgs = msgs_sorted[i:i + chunk_size]
            
            # Combine messages in this chunk
            combined_messages = []
            timestamps = []
            for msg in chunk_msgs:
                message_text = msg.get("message", "")
                timestamp = msg.get("timestamp", "")
                combined_messages.append(f"- {message_text}")
                timestamps.append(timestamp)
            
            # Create enriched content
            content = f"{user_name}'s messages:\n" + "\n".join(combined_messages)
            
            # Metadata
            metadata = {
                "user_name": user_name,
                "user_id": msgs[0].get("user_id", ""),
                "timestamp": timestamps[0] if timestamps else "",
                "timestamp_range": f"{timestamps[0]} to {timestamps[-1]}" if len(timestamps) > 1 else timestamps[0] if timestamps else "",
                "message_count": len(chunk_msgs),
                "doc_type": "aggregated",
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
    
    logger.info(f"Created {len(documents)} aggregated documents from {len(user_messages)} users")
    return documents


def build_faiss_index(
    documents: List[Document],
    embeddings: OpenAIEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from documents.
    
    Args:
        documents: List of Document objects
        embeddings: OpenAI embeddings model
        
    Returns:
        FAISS vector store
    """
    logger.info(f"Building FAISS index from {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    logger.info("FAISS index built successfully")
    return vectorstore


def save_faiss_index(vectorstore: FAISS, index_dir: str = "./faiss_index"):
    """
    Save FAISS index to disk.
    
    Args:
        vectorstore: FAISS vector store to save
        index_dir: Directory to save the index
    """
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info(f"FAISS index saved to {index_dir}")


def load_faiss_index(
    embeddings: OpenAIEmbeddings,
    index_dir: str = "./faiss_index"
) -> Optional[FAISS]:
    """
    Load FAISS index from disk if it exists.
    
    Args:
        embeddings: OpenAI embeddings model
        index_dir: Directory where the index is saved
        
    Returns:
        FAISS vector store or None if not found
    """
    index_path = Path(index_dir)
    if not index_path.exists():
        logger.info(f"No existing FAISS index found at {index_dir}")
        return None
    
    try:
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {index_dir}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None


def semantic_search(
    vectorstore: FAISS,
    query: str,
    k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search on the FAISS index.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of top results to return
        
    Returns:
        List of tuples (Document, similarity_score)
    """
    logger.info(f"Performing semantic search for: '{query}' (k={k})")
    results = vectorstore.similarity_search_with_score(query, k=k)
    logger.info(f"Retrieved {len(results)} documents")
    return results


def format_retrieved_context(top_docs: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        top_docs: List of tuples (Document, similarity_score)
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, (doc, score) in enumerate(top_docs, 1):
        user_name = doc.metadata.get("user_name", "Unknown")
        message = doc.metadata.get("message", doc.page_content)
        timestamp = doc.metadata.get("timestamp", "unknown time")
        
        context_parts.append(
            f"[{i}] {user_name} (at {timestamp}) [relevance: {score:.4f}]:\n{message}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    return context

