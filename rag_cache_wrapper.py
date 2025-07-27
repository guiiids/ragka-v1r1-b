"""
RAG Cache Wrapper for RAGKA

This module provides a wrapper around the RAG assistant that adds caching functionality.
It intercepts calls to the RAG assistant and checks the cache before delegating to the wrapped instance.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from services.redis_service import redis_service

# Configure logging
logger = logging.getLogger(__name__)

class RagCacheWrapper:
    """
    Wrapper around the RAG assistant that adds caching functionality.
    
    This wrapper intercepts calls to the RAG assistant and checks the cache
    before delegating to the wrapped instance. It caches:
    
    1. Embeddings: Generated embeddings for queries and documents
    2. Search Results: Results from the knowledge base search
    3. RAG Responses: Generated responses for user queries
    
    Follow-up questions are not cached because they depend on the conversation history.
    """
    
    def __init__(self, rag_assistant):
        """
        Initialize the RAG cache wrapper.
        
        Args:
            rag_assistant: The RAG assistant to wrap
        """
        self.rag_assistant = rag_assistant
        self.cache_enabled = redis_service.is_connected()
        
        # Cache expiration times (in seconds)
        self.embedding_expiration = 604800  # 7 days
        self.search_expiration = 86400     # 24 hours
        self.response_expiration = 43200   # 12 hours
        
        # Cache key prefixes
        self.embedding_prefix = "ragka:embedding:"
        self.search_prefix = "ragka:search:"
        self.response_prefix = "ragka:response:"
        
        logger.info(f"RAG cache wrapper initialized. Cache enabled: {self.cache_enabled}")
    
    def _generate_cache_key(self, prefix: str, data: str) -> str:
        """
        Generate a cache key for the given data.
        
        Args:
            prefix: The cache key prefix
            data: The data to generate a key for
            
        Returns:
            The cache key
        """
        # Generate an MD5 hash of the data
        hash_obj = hashlib.md5(data.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Return the key with the prefix
        return f"{prefix}{hash_hex}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text, with caching.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            The embedding
        """
        if not self.cache_enabled:
            return self.rag_assistant.generate_embedding(text)
        
        # Generate a cache key
        cache_key = self._generate_cache_key(self.embedding_prefix, text)
        
        # Check if the embedding is in the cache
        cached_embedding = redis_service.get(cache_key)
        if cached_embedding is not None:
            logger.debug(f"Cache hit for embedding: {text[:50]}...")
            return cached_embedding
        
        # Generate the embedding
        logger.debug(f"Cache miss for embedding: {text[:50]}...")
        embedding = self.rag_assistant.generate_embedding(text)
        
        # Cache the embedding
        redis_service.set(cache_key, embedding, self.embedding_expiration)
        
        return embedding
    
    def search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for the given query, with caching.
        
        Args:
            query: The query to search for
            
        Returns:
            The search results
        """
        if not self.cache_enabled:
            return self.rag_assistant.search_knowledge_base(query)
        
        # Generate a cache key
        cache_key = self._generate_cache_key(self.search_prefix, query)
        
        # Check if the search results are in the cache
        cached_results = redis_service.get(cache_key)
        if cached_results is not None:
            logger.debug(f"Cache hit for search: {query}")
            return cached_results
        
        # Search the knowledge base
        logger.debug(f"Cache miss for search: {query}")
        results = self.rag_assistant.search_knowledge_base(query)
        
        # Cache the search results
        redis_service.set(cache_key, results, self.search_expiration)
        
        return results
    
    def generate_rag_response(self, query: str, is_enhanced: bool = False) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
        """
        Generate a RAG response for the given query, with caching.
        
        Args:
            query: The query to generate a response for
            is_enhanced: Whether to use enhanced RAG
            
        Returns:
            A tuple of (answer, cited_sources, uncited_sources, evaluation, context)
        """
        # Check if this is a follow-up question
        history = self.rag_assistant.conversation_manager.get_history()
        is_follow_up = len(history) > 2  # More than system message + 1 turn
        
        # Don't cache follow-up questions
        if is_follow_up or not self.cache_enabled:
            logger.debug(f"Not caching follow-up question: {query}" if is_follow_up else "Cache disabled")
            return self.rag_assistant.generate_rag_response(query, is_enhanced)
        
        # Generate a cache key
        cache_key = self._generate_cache_key(self.response_prefix, f"{query}_{is_enhanced}")
        
        # Check if the response is in the cache
        cached_response = redis_service.get(cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for response: {query}")
            
            # Update conversation history
            self.rag_assistant.conversation_manager.add_user_message(query)
            self.rag_assistant.conversation_manager.add_assistant_message(cached_response["answer"])
            
            # Return the cached response
            return (
                cached_response["answer"],
                cached_response["cited_sources"],
                [],  # Uncited sources are not cached
                cached_response["evaluation"],
                cached_response["context"]
            )
        
        # Generate the response
        logger.debug(f"Cache miss for response: {query}")
        answer, cited_sources, uncited_sources, evaluation, context = self.rag_assistant.generate_rag_response(query, is_enhanced)
        
        # Cache the response
        cached_response = {
            "answer": answer,
            "cited_sources": cited_sources,
            "evaluation": evaluation,
            "context": context,
            "timestamp": time.time()
        }
        redis_service.set(cache_key, cached_response, self.response_expiration)
        
        return answer, cited_sources, uncited_sources, evaluation, context
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.rag_assistant.conversation_manager.clear_history()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_enabled:
            return {"enabled": False}
        
        # Get Redis health information
        health = redis_service.health_check()
        
        # Return cache statistics
        return {
            "enabled": True,
            "connected": health.get("connected", False),
            "mode": health.get("mode", "unknown"),
            "version": health.get("version", "unknown"),
            "used_memory": health.get("used_memory", "unknown"),
            "clients_connected": health.get("clients_connected", 0),
            "uptime_seconds": health.get("uptime_seconds", 0)
        }
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear the cache.
        
        Args:
            cache_type: The type of cache to clear (embedding, search, response, or None for all)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_enabled:
            return False
        
        try:
            if cache_type == "embedding":
                redis_service.delete_pattern(f"{self.embedding_prefix}*")
            elif cache_type == "search":
                redis_service.delete_pattern(f"{self.search_prefix}*")
            elif cache_type == "response":
                redis_service.delete_pattern(f"{self.response_prefix}*")
            else:
                # Clear all cache types
                redis_service.delete_pattern("ragka:*")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    # Delegate all other methods to the wrapped RAG assistant
    def __getattr__(self, name):
        """
        Delegate all other methods to the wrapped RAG assistant.
        
        Args:
            name: The name of the method to delegate
            
        Returns:
            The method from the wrapped RAG assistant
        """
        return getattr(self.rag_assistant, name)
