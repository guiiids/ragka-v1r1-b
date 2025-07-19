"""
Query mediator using LLM to detect contextual queries that need search.
"""
import logging
import json
from typing import Dict, List, Optional
from gpt4_intent_classifier import GPT4IntentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryMediatorLLM(GPT4IntentClassifier):
    """Mediator that detects when contextual queries need search."""
    
    def __init__(self, **kwargs):
        """Initialize the mediator with parent GPT4IntentClassifier settings."""
        super().__init__(**kwargs)
        
        # Override valid types for mediator
        self.valid_types = {
            'CONTEXTUAL_FOLLOW_UP',      # Pure context, no search needed
            'CONTEXTUAL_WITH_SEARCH',    # Context + new entities needing search
            'NEW_TOPIC'                  # Fresh topic needing search
        }
        
        # Conservative threshold - only activate when very uncertain
        self.activation_threshold = 0.3
        
    def should_mediate(self, current_confidence: float) -> bool:
        """Determine if mediator should activate based on current system confidence."""
        return current_confidence < self.activation_threshold
    
    def mediate_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        current_confidence: float = 1.0
    ) -> Dict:
        """
        Analyze query to determine if it needs search despite being contextual.
        
        Args:
            query: The user's query
            conversation_history: Optional list of previous conversation turns
            current_confidence: Confidence score from existing classifier
            
        Returns:
            Dict containing classification decision and reasoning
        """
        # Skip mediation if current system is confident
        if not self.should_mediate(current_confidence):
            return {
                'source': 'existing',
                'needs_mediation': False,
                'reason': f'Current confidence {current_confidence:.2f} above threshold'
            }
        
        try:
            # Format conversation context
            context = self._format_history(conversation_history) if conversation_history else "No previous context"
            
            # Construct the analysis prompt
            prompt = f"""You are a query classification mediator. Your job is to determine if a user query needs search or can be answered from conversation context alone.

CONVERSATION CONTEXT:
{context}

CURRENT QUERY: "{query}"

Analyze this query and respond with JSON:
{{
    "classification": "CONTEXTUAL_FOLLOW_UP" | "CONTEXTUAL_WITH_SEARCH" | "NEW_TOPIC",
    "needs_search": true/false,
    "confidence": 0.0-1.0,
    "external_entities": ["entity1", "entity2"],
    "reasoning": "Brief explanation of your decision"
}}

Key rules:
- CONTEXTUAL_FOLLOW_UP: Can be answered using only the conversation context
- CONTEXTUAL_WITH_SEARCH: Appears contextual but introduces new entities/concepts that need search
- NEW_TOPIC: Completely new topic requiring search
- external_entities: Any new products, tools, concepts, part numbers, etc. not mentioned in recent context
- Focus especially on comparisons, compatibility questions, and references to external systems

Examples:
- "What is iLab?" → NEW_TOPIC, needs_search: true
- "How do I use it?" → CONTEXTUAL_FOLLOW_UP, needs_search: false  
- "Is it the same as OpenLab?" → CONTEXTUAL_WITH_SEARCH, needs_search: true, external_entities: ["OpenLab"]
- "Does it work with Docker?" → CONTEXTUAL_WITH_SEARCH, needs_search: true, external_entities: ["Docker"]
- "What's the part number for v2.1?" → CONTEXTUAL_WITH_SEARCH, needs_search: true, external_entities: ["v2.1"]"""

            # Get LLM response
            response = self.openai_client.get_chat_response(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.1  # Keep it deterministic
            )
            
            # Parse JSON response
            try:
                result = json.loads(response.strip())
                
                # Validate required fields
                required_fields = ['classification', 'needs_search', 'confidence', 'reasoning']
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                # Ensure external_entities is a list
                if 'external_entities' not in result:
                    result['external_entities'] = []
                
                logger.info(
                    f"Mediator analyzed query '{query}': "
                    f"{result['classification']}, "
                    f"search_needed={result['needs_search']}, "
                    f"confidence={result['confidence']:.2f}, "
                    f"entities={result['external_entities']}"
                )
                
                return {
                    'source': 'mediator',
                    'needs_mediation': True,
                    'decision': result,
                    'original_confidence': current_confidence
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse mediator response: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Mediation failed: {e}")
            # Fall back to treating it as a new topic
            return {
                'source': 'mediator_fallback',
                'needs_mediation': True,
                'decision': {
                    'classification': 'NEW_TOPIC',
                    'needs_search': True,
                    'confidence': 0.5,
                    'external_entities': [],
                    'reasoning': f'Mediation failed: {str(e)}'
                },
                'original_confidence': current_confidence
            }
