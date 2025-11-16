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
    
    def __init__(self, llm, embeddings, api_url: str, index_dir: str = "./faiss_index", k: int = 3, doc_strategy: str = "individual"):
        """
        Initialize RAG nodes.
        
        Args:
            llm: Language model instance
            embeddings: OpenAI embeddings instance
            api_url: URL for the messages API
            index_dir: Directory to save/load FAISS index
            k: Number of documents to retrieve (default: 3, OPTIMIZED for 512MB memory)
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
        
        # Expand query for counting/enumeration questions (OPTIMIZED: max 2 queries)
        expanded_queries = self._expand_query(question)
        
        # Perform semantic search with all queries
        all_docs = {}  # Use dict to deduplicate by content
        for query in expanded_queries:
            results = semantic_search(self._vectorstore, query, k=self.k)
            for doc, score in results:
                content = doc.page_content
                if content not in all_docs or all_docs[content][1] > score:
                    all_docs[content] = (doc, score)
        
        # Convert back to list and sort by score (OPTIMIZED: limit to k docs only)
        top_docs = sorted(all_docs.values(), key=lambda x: x[1])[:self.k]
        
        # Extract potential user names from question for hybrid retrieval
        extracted_names = self._extract_names_from_question(question)
        
        # If we detected a name, boost results from that user (OPTIMIZED: limited boosting)
        if extracted_names:
            logger.info(f"Detected name(s) in question: {extracted_names}")
            top_docs = self._boost_user_documents(top_docs, extracted_names)
        
        # Format context for LLM (memory-optimized)
        context = format_retrieved_context(top_docs)
        
        # Store only essential data to reduce memory
        state["top_docs"] = top_docs
        state["relevant_context"] = context
        
        logger.info(f"Retrieved {len(top_docs)} relevant documents")
        return state
    
    def _expand_query(self, question: str) -> List[str]:
        """
        Expand the query for better coverage across different question types.
        
        OPTIMIZED FOR MEMORY: Limits to max 2-3 queries to reduce semantic search calls.
        
        Handles:
        - Counting questions: "How many cars does X have?" -> ["original", "X cars"]
        - Who questions: "Who has travel plans?" -> ["original", "travel vacation trip"]
        - What questions with names: "What are X's restaurants?" -> ["original", "X restaurants dining"]
        """
        queries = [question]
        
        question_lower = question.lower()
        
        # Detect question type
        is_who_question = question_lower.startswith('who ')
        is_what_question = question_lower.startswith('what ')
        is_counting = any(keyword in question_lower for keyword in ['how many', 'count', 'number of', 'list all', 'what are all'])
        
        # Detect the subject/topic with expanded keywords
        subject_keywords = {
            'travel': ['travel', 'trip', 'trips', 'vacation', 'journey', 'visit', 'visiting'],
            'car': ['car', 'cars', 'vehicle', 'vehicles', 'BMW', 'Mercedes', 'Tesla', 'automobile'],
            'restaurant': ['restaurant', 'restaurants', 'dining', 'food', 'Italian', 'cuisine', 'eatery'],
            'hotel': ['hotel', 'hotels', 'accommodation', 'stay', 'staying'],
        }
        
        detected_topics = []
        for topic, keywords in subject_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_topics.append((topic, keywords))
        
        # Extract names from question
        names = self._extract_names_from_question(question)
        
        # OPTIMIZED: Add ONE most relevant expanded query based on question type
        if is_who_question and detected_topics:
            # For "Who has X?" questions, expand with topic synonyms
            topic, keywords = detected_topics[0]
            # Use top 3 most relevant synonyms
            synonym_query = ' '.join(keywords[:3])
            queries.append(synonym_query)
        elif is_counting and names and detected_topics:
            # For counting with name: "How many cars does X have?"
            name = names[0]
            topic = detected_topics[0][0]
            queries.append(f"{name} {topic}")
        elif names and detected_topics:
            # For name-based questions: "What are X's restaurants?"
            name = names[0]
            topic = detected_topics[0][0]
            # Add name + topic keywords
            topic_keywords = detected_topics[0][1][:2]  # Use top 2 keywords
            queries.append(f"{name} {' '.join(topic_keywords)}")
        
        logger.info(f"Expanded query to {len(queries)} variants: {queries}")
        return queries
    
    def _extract_names_from_question(self, question: str) -> List[str]:
        """
        Extract potential user names from the question.
        
        Enhanced heuristic: looks for capitalized words and common name patterns.
        """
        import re
        # Look for capitalized words (simple name detection)
        words = question.split()
        names = []
        
        # Common question words and pronouns to skip
        skip_words = {
            'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose',
            'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'it', 'its', 'they', 'them', 'their', 'theirs',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        }
        
        for word in words:
            # Remove punctuation and check if it's capitalized
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and len(clean_word) > 1:
                # Check if first letter is uppercase
                if clean_word[0].isupper():
                    # Skip common question words
                    if clean_word.lower() not in skip_words:
                        names.append(clean_word)
        
        logger.info(f"Extracted potential names from question: {names}")
        return names
    
    def _boost_user_documents(self, top_docs: List[Tuple[Document, float]], names: List[str]) -> List[Tuple[Document, float]]:
        """
        Boost documents from specific users by retrieving more of their messages.
        
        OPTIMIZED FOR MEMORY: Limits additional searches and result size.
        Handles name variations for better matching.
        """
        # OPTIMIZED: Only process first name to reduce memory usage
        if not names:
            return top_docs
        
        name = names[0]  # Only boost for the first detected name
        
        # Get additional documents from the specific user (OPTIMIZED: fetch fewer)
        enhanced_docs = list(top_docs)
        
        # Search for more documents from this user with multiple name variations
        # For names like "Amira", also try "Amina" or similar
        filter_query = f"{name} says messages"
        additional_docs = semantic_search(self._vectorstore, filter_query, k=max(3, self.k // 2))
        
        # Add documents that are from the target user and not already in top_docs
        existing_contents = {doc.page_content for doc, _ in enhanced_docs}
        for doc, score in additional_docs:
            doc_user = doc.metadata.get("user_name", "")
            # Check for name match with fuzzy matching (handles variations)
            if self._name_matches(name, doc_user) and doc.page_content not in existing_contents:
                # Boost the score slightly for user-matched documents
                enhanced_docs.append((doc, score * 0.95))  # Slight boost
                existing_contents.add(doc.page_content)
        
        # Sort by score (lower is better in FAISS distance metric)
        enhanced_docs.sort(key=lambda x: x[1])
        
        # OPTIMIZED: Return only k docs (not k*1.5) to save memory
        return enhanced_docs[:self.k]
    
    def _name_matches(self, query_name: str, doc_user_name: str) -> bool:
        """
        Check if a query name matches a document user name.
        
        Handles variations like:
        - Exact match: "Layla" matches "Layla Kawaguchi"
        - Partial match: "Amira" might match "Amina" (if close enough)
        - Case insensitive
        """
        query_lower = query_name.lower()
        doc_lower = doc_user_name.lower()
        
        # Direct substring match
        if query_lower in doc_lower:
            return True
        
        # Check if the query name is similar to any part of the doc name
        # Simple similarity: if 80%+ of characters match in order
        doc_parts = doc_lower.split()
        for part in doc_parts:
            if len(query_lower) >= 3 and len(part) >= 3:
                # Calculate simple character overlap
                matching_chars = sum(1 for i, c in enumerate(query_lower) 
                                   if i < len(part) and c == part[i])
                similarity = matching_chars / max(len(query_lower), len(part))
                if similarity >= 0.75:  # 75% similarity threshold
                    logger.info(f"Fuzzy name match: '{query_name}' ~= '{part}' in '{doc_user_name}'")
                    return True
        
        return False
    
    def generate_answer(self, state: AgentState) -> AgentState:
        """
        Node: Generate answer using LLM with retrieved context.
        
        OPTIMIZED FOR MEMORY: Cleans up state after generation.
        
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
- For date/time questions, extract specific dates, days of the week, or time periods mentioned in messages. If a timestamp is provided, use it to give context like "in March 2024" or specific dates.
- For counting questions (how many, count, list), count ALL unique items mentioned across ALL messages provided.
- When asked "how many X does Y have?", interpret this as "how many X are mentioned by/about Y?"
  Examples: "how many cars" = count all car types/brands mentioned (BMW, Tesla, Mercedes, etc.)
- For aggregation questions like "Who has travel plans?" or "Who likes X?", review EVERY message in the context and list ALL members mentioned. Be exhaustive.
- Some messages are aggregated (grouped together) and some are individual - use all of them.
- If you see the same information repeated across multiple messages, count unique items only once.
- When listing people or items, be thorough and check all messages before responding.
- For name-based queries, consider common name variations (e.g., Amira might be Amina, or vice versa) - if you find a similar name, mention it.
- When counting or listing, show your work by mentioning what you found to be transparent.
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
        
        # OPTIMIZED: Don't store full conversation history to save memory
        # Only keep minimal message history
        state["messages"] = [
            HumanMessage(content=question),
            AIMessage(content=answer)
        ]
        
        # OPTIMIZED: Clear large objects from state to free memory
        state["top_docs"] = []  # Clear document references
        state["relevant_context"] = ""  # Clear context string
        
        logger.info(f"Generated answer: {answer[:100]}...")
        return state
    
    def clear_cache(self):
        """Clear the vectorstore cache to force rebuild."""
        self._vectorstore = None
        logger.info("Vectorstore cache cleared")

