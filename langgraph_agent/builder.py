"""
LangGraph agent builder for the RAG-based QnA system.

This module constructs the RAG workflow using FAISS and semantic search.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START

from langgraph_agent.state import AgentState
from langgraph_agent.nodes import RAGNodes
from langgraph_agent.utils import MESSAGES_API_URL

load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class QnAAgent:
    """
    RAG-based QnA agent for answering questions about member data.
    
    This agent uses:
    - FAISS for vector storage and semantic search
    - OpenAI embeddings (text-embedding-3-small)
    - LangGraph for workflow orchestration
    - Semantic retrieval instead of name-based lookup
    
    The workflow consists of three nodes:
    1. Load & Index: Fetches data and builds/loads FAISS index
    2. Retrieve Context: Performs semantic search to find relevant messages
    3. Generate Answer: Produces answer using LLM with retrieved context
    
    Advantages over name-based approach:
    - Can answer content-based questions ("Who likes Italian restaurants?")
    - Retrieves only relevant messages, not entire member history
    - Works with questions that don't mention specific names
    - Better context quality with semantic similarity scoring
    """

    def __init__(
        self,
        api_url: str = MESSAGES_API_URL,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None,
        index_dir: str = "./faiss_index",
        k: int = 10,
        doc_strategy: str = "hybrid",
    ):
        """
        Initialize the RAG-based QnA agent.
        
        Args:
            api_url: URL of the messages API
            llm_model: Name of the OpenAI LLM model (default: gpt-4o-mini)
            embedding_model: Name of the OpenAI embedding model (default: text-embedding-3-small)
            openai_api_key: OpenAI API key (defaults to env var)
            index_dir: Directory to save/load FAISS index
            k: Number of documents to retrieve per query (default: 10 for better coverage)
            doc_strategy: Document creation strategy - "individual", "aggregated", or "hybrid" (default: hybrid)
        """
        self.api_url = api_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        self.k = k
        self.doc_strategy = doc_strategy
        
        api_key = openai_api_key or OPENAI_API_KEY
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            temperature=0.1,
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )
        
        # Initialize RAG nodes
        self.nodes = RAGNodes(
            llm=self.llm,
            embeddings=self.embeddings,
            api_url=self.api_url,
            index_dir=self.index_dir,
            k=self.k,
            doc_strategy=self.doc_strategy,
        )
        
        # Build the graph
        self._graph = self._build_graph()
        
        logger.info(f"RAG Agent initialized with {llm_model} and {embedding_model}")

    def _build_graph(self):
        """
        Build the RAG workflow using LangGraph.
        
        Workflow:
        START -> load_and_index -> retrieve_context -> generate_answer -> END
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("load_and_index", self.nodes.load_and_index)
        workflow.add_node("retrieve_context", self.nodes.retrieve_context)
        workflow.add_node("generate_answer", self.nodes.generate_answer)
        
        # Define edges (workflow flow)
        workflow.add_edge(START, "load_and_index")
        workflow.add_edge("load_and_index", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        logger.info("RAG workflow built successfully")
        return workflow.compile()

    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer using RAG.
        
        The system will:
        1. Load/build FAISS index (if needed)
        2. Perform semantic search to find relevant messages
        3. Generate answer based on retrieved context
        
        Args:
            question: Natural language question about member data
            
        Returns:
            Answer string
        """
        initial_state: AgentState = {
            "messages": [],
            "question": question,
            "top_docs": [],
            "relevant_context": "",
            "answer": "",
        }
        
        logger.info(f"Processing question with RAG: {question}")
        final_state = self._graph.invoke(initial_state)
        
        return final_state["answer"]

    def clear_cache(self):
        """
        Clear the FAISS vectorstore cache to force rebuild.
        
        Use this when the member data has been updated and you want
        to rebuild the index from the latest API data.
        """
        self.nodes.clear_cache()
        logger.info("RAG cache cleared")


def create_agent(
    api_url: str = MESSAGES_API_URL,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    k: int = 10,
    doc_strategy: str = "hybrid",
) -> QnAAgent:
    """
    Factory function to create a RAG-based QnA agent instance.
    
    Args:
        api_url: URL of the messages API
        llm_model: Name of the OpenAI LLM model
        embedding_model: Name of the OpenAI embedding model
        k: Number of documents to retrieve per query (default: 10)
        doc_strategy: Document creation strategy (default: hybrid)
        
    Returns:
        Initialized RAG QnA agent
    """
    return QnAAgent(
        api_url=api_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        k=k,
        doc_strategy=doc_strategy,
    )

