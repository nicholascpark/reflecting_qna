"""
State schema for the QnA LangGraph agent with RAG support.
"""

from typing import TypedDict, List, Dict, Annotated, Tuple
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


class AgentState(TypedDict):
    """
    State for the RAG-based QnA agent.
    
    Attributes:
        messages: Conversation history (uses add_messages for automatic merging)
        question: Current question being processed
        top_docs: Top retrieved documents with similarity scores
        relevant_context: Relevant member messages extracted via semantic search
        answer: Generated answer for the question
    """
    messages: Annotated[List, add_messages]  # Conversation history
    question: str  # Current question
    top_docs: List[Tuple[Document, float]]  # Retrieved documents with scores
    relevant_context: str  # Context built from retrieved documents
    answer: str  # Generated answer

