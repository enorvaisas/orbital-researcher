"""Define the configurable parameters for the research agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, TypedDict, Literal
from enum import Enum

from langchain_core.runnables import RunnableConfig, ensure_config

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"

class ConfigSchema(TypedDict):
    """Schema for the agent configuration."""
    system_prompt: str
    model: str
    max_search_results: int
    max_web_research_loops: int
    search_api: str
    fetch_full_page: bool

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default="You are a research assistant that helps people find information and analyze it.",
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    
    max_web_research_loops: int = field(
        default=3,
        metadata={
            "description": "Number of research iterations to perform for deep dive research."
        }
    )
    
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = field(
        default="duckduckgo",
        metadata={
            "description": "Web search API to use for research."
        }
    )
    
    fetch_full_page: bool = field(
        default=True,
        metadata={
            "description": "Whether to fetch and analyze the full content of web pages."
        }
    )
    
    ollama_base_url: str = field(
        default="http://localhost:11434/",
        metadata={
            "description": "Base URL for Ollama API."
        }
    )
    
    lmstudio_base_url: str = field(
        default="http://localhost:1234/v1",
        metadata={
            "description": "Base URL for LMStudio OpenAI-compatible API."
        }
    )
    
    llm_provider: Literal["openai", "anthropic", "ollama", "lmstudio"] = field(
        default="anthropic",
        metadata={
            "description": "Provider for the LLM (OpenAI, Anthropic, Ollama, or LMStudio)."
        }
    )
    
    strip_thinking_tokens: bool = field(
        default=True,
        metadata={
            "description": "Whether to strip <think> tokens from model responses."
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields}) 