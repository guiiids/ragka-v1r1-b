"""
OpenAIService class for handling interactions with the Azure OpenAI API
"""
import logging
from openai import AzureOpenAI
from openai_logger import log_openai_call

logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Handles interactions with the Azure OpenAI API.
    
    This class is responsible for:
    - Initializing the Azure OpenAI client
    - Sending requests to the API
    - Processing responses
    - Error handling and logging
    """
    
    def __init__(self, azure_endpoint=None, api_key=None, api_version="2024-02-01", deployment_name=None):
        """
        Initialize the OpenAI service.
        
        Args:
            azure_endpoint: The Azure OpenAI endpoint URL
            api_key: The API key for authentication
            api_version: The API version to use
            deployment_name: The deployment name to use for chat completions
        """
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.deployment_name = deployment_name
        
        # Initialize the OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        logger.debug(f"OpenAIService initialized with endpoint: {azure_endpoint}, api_version: {api_version}, deployment: {deployment_name}")
    
    def get_chat_response(
        self,
        messages,
        temperature=0.3,
        max_tokens=1000,
        max_completion_tokens=None,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ):
        """
        Get a response from the OpenAI chat completions API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens for standard models
            max_completion_tokens: Maximum tokens for models that use the
                ``max_completion_tokens`` parameter (e.g., ``o4-mini``)
            top_p: Controls diversity via nucleus sampling
            presence_penalty: Penalizes new tokens based on presence in text so far
            frequency_penalty: Penalizes new tokens based on frequency in text so far
            
        Returns:
            The assistant's response text
        """
        logger.info(f"Sending request to OpenAI with {len(messages)} messages")
        logger.debug(
            f"Using temperature: {temperature}, max_tokens: {max_tokens}, max_completion_tokens: {max_completion_tokens}, top_p: {top_p}"
        )
        
        try:
            # Prepare the request
            request = {
                'model': self.deployment_name,
                'messages': messages,
                'temperature': temperature,
                'top_p': top_p,
                'presence_penalty': presence_penalty,
                'frequency_penalty': frequency_penalty
            }

            if max_completion_tokens is not None:
                request['max_completion_tokens'] = max_completion_tokens
            else:
                request['max_tokens'] = max_tokens
            
            # Log the first and last message for debugging
            if messages:
                logger.debug(f"First message - Role: {messages[0]['role']}")
                logger.debug(f"Last message - Role: {messages[-1]['role']}")
            
            # Send the request to the API
            response = self.client.chat.completions.create(**request)
            
            # Log the API call
            log_openai_call(request, response)
            
            # Extract and return the response text
            answer = response.choices[0].message.content
            logger.info(f"Received response from OpenAI (length: {len(answer)})")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Re-raise the exception to be handled by the caller
            raise