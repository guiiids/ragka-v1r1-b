"""
QueryMediator class for providing LLM-based query classification when existing classifier is uncertain
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Set

# Configure logging
logger = logging.getLogger(__name__)

class QueryMediator:
    """
    A conservative mediator that uses LLM to classify queries when the existing classifier
    has low confidence. Only activates when existing classifier confidence is very low.
    """
    
    def __init__(self, openai_service, confidence_threshold: float = 0.6):
        """
        Initialize the query mediator.
        
        Args:
            openai_service: OpenAIService instance for LLM interactions
            confidence_threshold: Threshold below which mediator activates (default: 0.6)
        """
        self.openai_service = openai_service
        self.confidence_threshold = confidence_threshold
        logger.info(f"QueryMediator initialized with confidence threshold: {confidence_threshold}")
    
    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract potential entities (proper nouns, technical terms, product names) from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Set of potential entities found in the text
        """
        entities = set()
        
        # Extract capitalized words (potential proper nouns)
        # Look for words that start with capital letter and are not at sentence start
        words = text.split()
        for i, word in enumerate(words):
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Skip if empty after cleaning
            if not clean_word:
                continue
                
            # Check if it's a capitalized word (but not sentence start)
            if (clean_word[0].isupper() and len(clean_word) > 1 and 
                (i > 0 or not text[0].isupper())):  # Not sentence start
                entities.add(clean_word)
        
        # Extract technical terms with specific patterns
        # Product names with versions (e.g., "OpenLab v2.1", "iLab 3.0")
        version_pattern = r'\b([A-Z][a-zA-Z]+(?:\s+v?\d+(?:\.\d+)*)?)\b'
        for match in re.finditer(version_pattern, text):
            entities.add(match.group(1))
        
        # Acronyms (2-5 uppercase letters)
        acronym_pattern = r'\b[A-Z]{2,5}\b'
        for match in re.finditer(acronym_pattern, text):
            entities.add(match.group())
        
        # Technical terms with common suffixes
        tech_suffixes = ['Lab', 'System', 'Platform', 'Tool', 'API', 'SDK', 'Framework']
        for suffix in tech_suffixes:
            pattern = rf'\b\w+{suffix}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.add(match.group())
        
        logger.debug(f"Extracted entities from '{text}': {entities}")
        return entities
    
    def _detect_contextual_references(self, text: str) -> Set[str]:
        """
        Detect pronouns and demonstratives that require context resolution.
        
        Args:
            text: The text to analyze
            
        Returns:
            Set of contextual reference indicators found in the text
        """
        contextual_refs = set()
        text_lower = text.lower()
        
        # Demonstrative pronouns requiring context
        demonstratives = ['it', 'this', 'that', 'these', 'those', 'them', 'they']
        for demo in demonstratives:
            if re.search(rf'\b{demo}\b', text_lower):
                contextual_refs.add(demo)
        
        # Other contextual indicators
        contextual_indicators = [
            'the one', 'the same', 'such', 'like that', 'similar',
            'above', 'below', 'mentioned', 'said', 'discussed'
        ]
        for indicator in contextual_indicators:
            if indicator in text_lower:
                contextual_refs.add(indicator)
        
        logger.debug(f"Detected contextual references in '{text}': {contextual_refs}")
        return contextual_refs
    
    def _extract_entities_from_history(self, history: List[Dict]) -> Set[str]:
        """
        Extract entities mentioned in recent conversation history.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Set of entities found in the conversation history
        """
        entities = set()
        
        # Look at recent messages (last 6 messages = 3 turns)
        recent_messages = history[-6:] if len(history) > 6 else history
        
        for msg in recent_messages:
            content = msg.get('content', '')
            entities.update(self._extract_entities(content))
        
        logger.debug(f"Extracted entities from history: {entities}")
        return entities
    
    def _has_new_entities(self, query: str, history: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Check if the query introduces new entities not mentioned in recent conversation.
        This version correctly handles queries with both contextual references and new entities.
        
        Args:
            query: The current user query
            history: List of previous conversation turns
            
        Returns:
            Tuple of (has_new_entities, list_of_new_entities)
        """
        query_entities = self._extract_entities(query)
        history_entities = self._extract_entities_from_history(history)
        
        # Find entities in query that are not in recent history
        new_entities = query_entities - history_entities
        
        # Filter out contextual references and common words from the "new" list.
        # This is the key change: we don't want pronouns to count as new entities.
        contextual_words = {'it', 'this', 'that', 'these', 'those', 'them', 'they', 
                            'one', 'same', 'above', 'below', 'mentioned', 'said', 'discussed'}
        common_words = {'What', 'How', 'When', 'Where', 'Why'}
        
        # Normalize to lowercase for filtering
        new_entities_filtered = {
            entity for entity in new_entities 
            if entity.lower() not in contextual_words and entity not in common_words
        }
        
        has_new = len(new_entities_filtered) > 0
        
        if has_new:
            logger.info(f"Found new entities in query: {new_entities_filtered}")
        else:
            logger.info("No new entities found in query.")
            
        return has_new, list(new_entities_filtered)
    
    def should_mediate(self, current_classification: Dict[str, float]) -> bool:
        """
        Determine if mediator should intervene based on current classification confidence.
        
        Args:
            current_classification: Dictionary of classification scores from existing system
            
        Returns:
            True if mediator should intervene, False otherwise
        """
        if not current_classification:
            return True
            
        # Get highest confidence score
        max_confidence = max(current_classification.values())
        should_mediate = max_confidence < self.confidence_threshold
        
        if should_mediate:
            logger.info(f"Mediating due to low confidence: {max_confidence:.2f}")
        
        return should_mediate
    
    def _build_classification_prompt(self, query: str, history: List[Dict]) -> str:
        """
        Build the prompt for the LLM classifier.
        
        Args:
            query: The current user query
            history: List of previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # Get last few turns for context
        recent_context = self._format_recent_context(history, max_turns=3)
        
        return f"""You are a query classification mediator. Your job is to determine if a user query needs search or can be answered from conversation context alone.

CONVERSATION CONTEXT:
{recent_context}

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
"""
    
    def _format_recent_context(self, history: List[Dict], max_turns: int = 3) -> str:
        """
        Format recent conversation history for the prompt.
        
        Args:
            history: List of conversation messages
            max_turns: Maximum number of turns to include
            
        Returns:
            Formatted conversation context string
        """
        if not history:
            return "[No previous conversation]"
        
        # Get last few exchanges
        recent = history[-(max_turns * 2):]  # Each turn = user + assistant
        
        context_lines = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)
    
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the LLM response into a structured format.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Parsed response dictionary
        """
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
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing mediator response: {e}")
            # Fallback to safe default
            return {
                'classification': 'NEW_TOPIC',
                'needs_search': True,
                'confidence': 0.5,
                'external_entities': [],
                'reasoning': f'Parse error: {str(e)}',
                'error': True
            }
    
    def classify(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        current_classification: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Classify a query using LLM-based analysis with enhanced entity detection.
        
        Args:
            query: The user's query
            history: Optional conversation history
            current_classification: Optional current classification scores
            
        Returns:
            Classification result dictionary
        """
        history = history or []
        
        # First, check for new entities using our local detection
        has_new_entities, new_entities = self._has_new_entities(query, history)
        
        # If we detect new entities, we should definitely search regardless of confidence
        if has_new_entities:
            logger.info(f"Detected new entities {new_entities}, forcing search")
            return {
                'source': 'mediator',
                'classification': 'CONTEXTUAL_WITH_SEARCH',
                'needs_search': True,
                'confidence': 0.8,
                'external_entities': new_entities,
                'reasoning': f'Detected new entities not in conversation context: {new_entities}',
                'entity_detection_triggered': True
            }
        
        # Check if we should mediate based on confidence
        if current_classification and not self.should_mediate(current_classification):
            logger.info("Skipping mediation - current confidence sufficient and no new entities")
            return {
                'source': 'existing',
                'classification': current_classification
            }
        
        try:
            # Build the classification prompt
            prompt = self._build_classification_prompt(query, history)
            
            # Get LLM response
            response = self.openai_service.get_chat_response(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200  # Keep responses focused
            )
            
            # Parse and validate response
            result = self._parse_response(response)
            
            # Add source information
            result['source'] = 'mediator'
            if current_classification:
                result['fallback_from'] = current_classification
            
            # Add entity detection info
            result['entity_detection_triggered'] = False
            
            logger.info(
                f"Mediator classification: {result['classification']}, "
                f"confidence: {result['confidence']:.2f}, "
                f"needs_search: {result['needs_search']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mediator classification: {e}")
            # Fallback to existing classification or safe default
            return {
                'source': 'error',
                'classification': current_classification or {'NEW_TOPIC': 1.0},
                'error': str(e),
                'entity_detection_triggered': False
            }
