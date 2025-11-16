"""
Node functions for the RAG-based LangGraph QnA agent.

Each node represents a step in the RAG workflow.
"""

import logging
from typing import Optional, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langgraph_agent.state import AgentState
from langgraph_agent.utils import (
    fetch_all_messages,
    messages_to_documents,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    semantic_search,
    format_retrieved_context,
)

logger = logging.getLogger(__name__)


class RAGNodes:
    """
    Container for RAG agent node functions.
    
    This class holds the node functions and manages the FAISS vector store.
    """
    
    def __init__(self, llm, embeddings, api_url: str, index_dir: str = "./faiss_index", k: int = 5, doc_strategy: str = "individual"):
        """
        Initialize RAG nodes.
        
        Args:
            llm: Language model instance
            embeddings: OpenAI embeddings instance
            api_url: URL for the messages API
            index_dir: Directory to save/load FAISS index
            k: Number of documents to retrieve (default: 5, optimized for memory)
            doc_strategy: Document creation strategy ("individual", "aggregated", or "hybrid")
                        Note: "hybrid" uses 2x memory, use "individual" for memory-constrained environments
        """
        self.llm = llm
        self.embeddings = embeddings
        self.api_url = api_url
        self.index_dir = index_dir
        self.k = k
        self.doc_strategy = doc_strategy
        self._vectorstore: Optional[FAISS] = None
    
    def load_and_index(self, state: AgentState) -> AgentState:
        """
        Node: Load member data and build/load FAISS index.
        
        This node:
        1. Tries to load existing FAISS index
        2. If not found, fetches data from API and builds new index
        3. Saves the index for future use
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state (no changes to state, but initializes vectorstore)
        """
        if self._vectorstore is None:
            # Try to load existing index
            logger.info("Attempting to load existing FAISS index...")
            self._vectorstore = load_faiss_index(self.embeddings, self.index_dir)
            
            # If no index exists, build one
            if self._vectorstore is None:
                logger.info(f"No index found. Building new FAISS index with '{self.doc_strategy}' strategy...")
                
                # Fetch messages from API
                messages = fetch_all_messages(self.api_url)
                
                # Convert to documents using the specified strategy
                documents = messages_to_documents(messages, strategy=self.doc_strategy)
                
                # Build FAISS index
                self._vectorstore = build_faiss_index(documents, self.embeddings)
                
                # Save for future use
                save_faiss_index(self._vectorstore, self.index_dir)
        else:
            logger.info("Using existing vectorstore in memory")
        
        return state
    
    def retrieve_context(self, state: AgentState) -> AgentState:
        """
        Node: Perform enhanced semantic search with hybrid retrieval.
        
        This node uses FAISS to find the most semantically similar messages.
        For name-based queries, it also performs metadata filtering to ensure
        all relevant messages from that person are included.
        
        Args:
            state: Current agent state with question
            
        Returns:
            Updated state with top_docs and relevant_context populated
        """
        if self._vectorstore is None:
            raise ValueError("Vectorstore not initialized. Run load_and_index first.")
        
        question = state["question"]
        
        # Expand query for counting/enumeration questions
        expanded_queries = self._expand_query(question)
        
        # Perform semantic search with all queries
        all_docs = {}  # Use dict to deduplicate by content
        for query in expanded_queries:
            results = semantic_search(self._vectorstore, query, k=self.k)
            for doc, score in results:
                content = doc.page_content
                if content not in all_docs or all_docs[content][1] > score:
                    all_docs[content] = (doc, score)
        
        # Convert back to list and sort by score
        top_docs = sorted(all_docs.values(), key=lambda x: x[1])[:self.k * 2]
        
        # Extract potential user names from question for hybrid retrieval
        extracted_names = self._extract_names_from_question(question)
        
        # If we detected a name, boost results from that user
        if extracted_names:
            logger.info(f"Detected name(s) in question: {extracted_names}")
            top_docs = self._boost_user_documents(top_docs, extracted_names)
        
        # Format context for LLM
        context = format_retrieved_context(top_docs)
        
        state["top_docs"] = top_docs
        state["relevant_context"] = context
        
        logger.info(f"Retrieved {len(top_docs)} relevant documents")
        return state
    
    def _expand_query(self, question: str) -> List[str]:
        """
        Expand the query for better coverage, especially for counting questions.
        
        For questions like "How many cars does X have?", we expand to:
        - The original question
        - "X's cars vehicles"
        - "X mentioned cars"
        etc.
        """
        queries = [question]
        
        question_lower = question.lower()
        
        # Detect counting/enumeration questions
        counting_keywords = ['how many', 'count', 'number of', 'list all', 'what are all']
        is_counting = any(keyword in question_lower for keyword in counting_keywords)
        
        # Detect the subject (cars, restaurants, etc.)
        subjects = {
            'car': ['cars', 'vehicles', 'car service', 'BMW', 'Mercedes', 'Tesla', 'Bentley', 'Aston Martin'],
            'restaurant': ['restaurants', 'dining', 'eateries', 'food'],
            'trip': ['trips', 'travel', 'vacation', 'journey'],
            'hotel': ['hotels', 'accommodations', 'stays'],
        }
        
        detected_subject = None
        for subject, keywords in subjects.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_subject = subject
                break
        
        # Extract names from question
        names = self._extract_names_from_question(question)
        
        # If it's a counting question with a name and subject, add expanded queries
        if is_counting and names and detected_subject:
            for name in names:
                queries.append(f"{name} {detected_subject}")
                queries.append(f"{name} mentions {detected_subject}")
                queries.append(f"{name}'s {detected_subject}")
                if detected_subject in subjects:
                    for keyword in subjects[detected_subject][:3]:
                        queries.append(f"{name} {keyword}")
        
        logger.info(f"Expanded query to {len(queries)} variants: {queries[:3]}...")
        return queries
    
    def _extract_names_from_question(self, question: str) -> List[str]:
        """
        Extract potential user names from the question.
        
        Simple heuristic: look for capitalized words that might be names.
        """
        import re
        # Look for capitalized words (simple name detection)
        words = question.split()
        names = []
        for word in words:
            # Remove punctuation and check if it's capitalized
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Common question words to skip
                if clean_word.lower() not in ['who', 'what', 'when', 'where', 'why', 'how', 'which']:
                    names.append(clean_word)
        return names
    
    def _boost_user_documents(self, top_docs: List[Tuple[Document, float]], names: List[str]) -> List[Tuple[Document, float]]:
        """
        Boost documents from specific users by retrieving more of their messages.
        
        This helps ensure we get ALL relevant information about a specific person.
        """
        # Get additional documents from the specific user(s)
        enhanced_docs = list(top_docs)
        
        for name in names:
            # Search for more documents from this user
            filter_query = f"{name}'s messages"
            additional_docs = semantic_search(self._vectorstore, filter_query, k=self.k)
            
            # Add documents that are from the target user and not already in top_docs
            existing_contents = {doc.page_content for doc, _ in enhanced_docs}
            for doc, score in additional_docs:
                doc_user = doc.metadata.get("user_name", "")
                if name.lower() in doc_user.lower() and doc.page_content not in existing_contents:
                    # Boost the score slightly for user-matched documents
                    enhanced_docs.append((doc, score * 0.95))  # Slight boost
                    existing_contents.add(doc.page_content)
        
        # Sort by score (lower is better in FAISS distance metric)
        enhanced_docs.sort(key=lambda x: x[1])
        
        # Return top k*1.5 to give LLM more context for counting/aggregation questions
        return enhanced_docs[:int(self.k * 1.5)]
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """
        Node: Generate answer using LLM with retrieved context.
        
        Args:
            state: Current agent state with question and relevant_context
            
        Returns:
            Updated state with answer and messages populated
        """
        question = state["question"]
        context = state["relevant_context"]
        
        system_prompt = """You are a helpful assistant that answers questions about member data.
Use ONLY the provided context (member messages) to answer questions.
The context includes messages retrieved via semantic search - they are the most relevant to the question.

IMPORTANT INSTRUCTIONS:
- Be specific and cite the member's name when providing information.
- For date/time questions, look at the timestamp information carefully.
- For counting questions (how many, count, list), count ALL unique items mentioned across ALL messages provided.
- When asked "how many X does Y have?", interpret this as "how many X are mentioned by/about Y?"
  Examples: "how many cars" = count all car types/brands mentioned (BMW, Tesla, Mercedes, etc.)
- For aggregation questions, look across ALL the provided messages to gather complete information.
- Some messages are aggregated (grouped together) and some are individual - use all of them.
- If you see the same information repeated across multiple messages, count unique items only once.
- When counting, list out what you're counting to be transparent and provide the total.
- If truly no relevant information exists in the context, say so honestly."""

        user_prompt = f"""Question: {question}

Context (Most Relevant Member Messages):
{context}

Please answer the question based on the context provided above. Be concise and accurate."""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        answer = response.content.strip()
        state["answer"] = answer
        
        # Add to conversation history
        state["messages"].extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        
        logger.info(f"Generated answer: {answer[:100]}...")
        return state
    
    def clear_cache(self):
        """Clear the vectorstore cache to force rebuild."""
        self._vectorstore = None
        logger.info("Vectorstore cache cleared")

