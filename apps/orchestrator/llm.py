"""
Modular LLM Provider Pattern for Multi-Agent Architecture.

This module provides a factory pattern for LLM providers, allowing easy
switching between different chat agents (DeepSeek, Gemini, etc.).
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This allows us to easily swap between different chat agents
    while maintaining a consistent interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the provider (e.g., 'DeepSeek', 'Gemini')."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass
    
    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        """
        Invoke the LLM asynchronously with retry logic.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            LLM response object
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform a health check on the LLM provider.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        pass


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek LLM provider using OpenAI-compatible API.
    
    Uses the OpenAI Python SDK with DeepSeek's base URL.
    Model: deepseek-chat (V3)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
        """
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
        self._model_name = "deepseek-chat"
        self._base_url = "https://api.deepseek.com"
        
        logger.info(f"Initializing DeepSeek provider with model: {self._model_name}")
        
        # Initialize LangChain ChatOpenAI with DeepSeek configuration
        self._llm = ChatOpenAI(
            model=self._model_name,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=0.7,
            timeout=60.0
        )
        
        logger.info("DeepSeek provider initialized successfully")
    
    @property
    def name(self) -> str:
        return "DeepSeek"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        """
        Invoke DeepSeek LLM with exponential backoff retry logic.
        
        Handles network issues and rate limits automatically.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            LLM response object
        """
        try:
            logger.debug(f"Invoking DeepSeek with {len(messages)} messages")
            response = await self._llm.ainvoke(messages)
            logger.debug("DeepSeek invocation successful")
            return response
        except Exception as e:
            logger.error(f"DeepSeek invocation failed: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """
        Perform a health check by making a simple test call.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            from langchain_core.messages import HumanMessage
            test_message = [HumanMessage(content="Say 'ok' if you can read this.")]
            response = await self.ainvoke(test_message)
            # Check if we got a valid response
            if hasattr(response, 'content') and response.content:
                logger.info("DeepSeek health check passed")
                return True
            return False
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {str(e)}")
            return False


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    Supports switching between different agents via ACTIVE_AGENT env var.
    """
    
    _providers: dict[str, LLMProvider] = {}
    _active_provider: Optional[LLMProvider] = None
    
    @classmethod
    def get_provider(cls, provider_name: Optional[str] = None) -> LLMProvider:
        """
        Get or create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider to use. If None, reads from ACTIVE_AGENT env var.
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider_name is not supported or required env vars are missing
        """
        # Determine which provider to use
        if provider_name is None:
            provider_name = os.getenv("ACTIVE_AGENT", "deepseek").lower()
        
        # Return cached provider if available
        if provider_name in cls._providers:
            return cls._providers[provider_name]
        
        # Create new provider instance
        if provider_name == "deepseek":
            provider = DeepSeekProvider()
        else:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: deepseek"
            )
        
        # Cache the provider
        cls._providers[provider_name] = provider
        cls._active_provider = provider
        
        logger.info(f"Created and cached {provider.name} provider")
        return provider
    
    @classmethod
    def get_active_provider(cls) -> LLMProvider:
        """
        Get the currently active provider based on ACTIVE_AGENT env var.
        
        Returns:
            LLMProvider instance
        """
        if cls._active_provider is None:
            cls._active_provider = cls.get_provider()
        return cls._active_provider
    
    @classmethod
    def get_provider_info(cls) -> dict[str, str]:
        """
        Get information about the active provider.
        
        Returns:
            Dictionary with provider name and model name
        """
        provider = cls.get_active_provider()
        return {
            "name": provider.name,
            "model": provider.model_name,
            "status": "online" if provider else "offline"
        }


# Convenience function for backward compatibility
def get_llm() -> BaseChatModel:
    """
    Get the active LLM instance (backward compatibility wrapper).
    
    This function maintains compatibility with existing code that expects
    a LangChain chat model directly.
    
    Returns:
        BaseChatModel instance from the active provider
    """
    provider = LLMProviderFactory.get_active_provider()
    # Return the underlying LangChain model
    return provider._llm

