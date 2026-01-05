"""
Base Agent class for the Agent Fleet infrastructure.

Provides a standardized interface for specialized agents (Job Scout, Matchmaker, Talent Strategist)
with externalized configuration and prompts.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import structlog
from langchain_core.messages import BaseMessage

from apps.orchestrator.llm import LLMProviderFactory

# Configure structlog for structured JSON logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class BaseAgent:
    """
    Base class for specialized AI agents in the Agent Fleet.

    Each agent:
    - Loads its configuration from agents.yaml
    - Loads its prompt from a text file
    - Initializes its own LLM provider with agent-specific settings
    - Provides a standardized interface for LLM invocation
    """

    def __init__(self, agent_name: str):
        """
        Initialize the agent with configuration and prompt.

        Args:
            agent_name: Name of the agent (e.g., "job_scout", "matchmaker", "talent_strategist")

        Raises:
            ValueError: If agent_name is not found in agents.yaml
            FileNotFoundError: If prompt file is not found
        """
        self.agent_name = agent_name
        self.config = self._load_agent_config(agent_name)
        self.prompt_file = self.config["prompt_file"]  # Store filename, not content
        self.llm_provider = self._initialize_llm_provider()

        logger.info(
            "agent_initialized",
            agent_name=agent_name,
            provider=self.llm_provider.name,
            model=self.llm_provider.model_name,
            temperature=self.config.get("temperature"),
        )

    def _load_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Load agent configuration from agents.yaml.

        Args:
            agent_name: Name of the agent to load

        Returns:
            Dictionary with agent configuration

        Raises:
            ValueError: If agent_name is not found in the configuration
        """
        # Get the path to agents.yaml (in apps/orchestrator/settings/)
        orchestrator_dir = Path(__file__).parent.parent
        settings_dir = orchestrator_dir / "settings"
        config_path = settings_dir / "agents.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if agent_name not in config:
            raise ValueError(
                f"Agent '{agent_name}' not found in agents.yaml. "
                f"Available agents: {list(config.keys())}"
            )

        agent_config = config[agent_name]
        logger.info("agent_config_loaded", agent_name=agent_name, config=agent_config)
        return agent_config

    def _load_prompt(self, prompt_file: str) -> str:
        """
        Load prompt text from a file.

        Args:
            prompt_file: Name of the prompt file (e.g., "job_scout.txt")

        Returns:
            Prompt text as a string

        Raises:
            FileNotFoundError: If prompt file is not found
        """
        # Get the path to prompts directory
        orchestrator_dir = Path(__file__).parent.parent
        prompts_dir = orchestrator_dir / "prompts"
        prompt_path = prompts_dir / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            prompt_text = f.read().strip()

        logger.info("prompt_loaded", prompt_file=prompt_file, prompt_length=len(prompt_text))
        return prompt_text

    def _initialize_llm_provider(self):
        """
        Initialize the LLM provider using the factory with agent-specific configuration.

        Returns:
            LLMProvider instance
        """
        provider_name = self.config.get("provider", "deepseek")
        model_name = self.config.get("model")
        temperature = self.config.get("temperature")

        provider = LLMProviderFactory.get_provider(
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
        )

        return provider

    async def invoke(self, messages: List[BaseMessage], system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> Any:
        """
        Invoke the LLM with messages, optionally overriding the system prompt.

        Args:
            messages: List of LangChain message objects
            system_prompt: Optional system prompt to override the default agent prompt.
                         If None, reloads and uses the agent's default prompt from file.
            max_tokens: Optional maximum tokens for the response (for controlling response length).

        Returns:
            LLM response object
        """
        # If system_prompt is provided, use it; otherwise reload the agent's default prompt from file
        # This ensures prompt changes are picked up immediately (no caching for development)
        if system_prompt is None:
            system_prompt = self._load_prompt(self.prompt_file)

        # Create a copy of messages to avoid modifying the input
        from langchain_core.messages import SystemMessage

        # Create a new list with copies of messages
        messages_copy = list(messages)

        # Ensure the first message is a SystemMessage with the prompt
        # If messages already start with a SystemMessage, replace it
        # Otherwise, prepend the system prompt
        if messages_copy and isinstance(messages_copy[0], SystemMessage):
            messages_copy[0] = SystemMessage(content=system_prompt)
        else:
            messages_copy.insert(0, SystemMessage(content=system_prompt))

        logger.info(
            "agent_invoking_llm",
            agent_name=self.agent_name,
            message_count=len(messages_copy),
            max_tokens=max_tokens,
        )

        # If max_tokens is specified, bind it to the LLM for this invocation
        if max_tokens is not None and hasattr(self.llm_provider, '_llm'):
            # Use bind() to create a temporary LLM instance with max_tokens
            bound_llm = self.llm_provider._llm.bind(max_tokens=max_tokens)
            response = await bound_llm.ainvoke(messages_copy)
        else:
            response = await self.llm_provider.ainvoke(messages_copy)

        logger.info(
            "agent_llm_response_received",
            agent_name=self.agent_name,
            response_length=len(response.content) if hasattr(response, "content") else 0,
        )

        return response

