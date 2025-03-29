"""Define the state classes for the research agent graph."""

import operator
from dataclasses import dataclass, field
from typing import Annotated, List, Dict, Any, Optional, Sequence

from langchain_core.messages import BaseMessage

@dataclass(kw_only=True)
class State:
    """The state for the research agent graph."""
    # Messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add] = field(default_factory=list)
    
    # Research-specific state
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)
    
    # Control flow state
    is_last_step: bool = field(default=False)
    research_mode: bool = field(default=False)
    
    # Store the sources for citation
    sources: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class InputState:
    """The input state for the graph."""
    messages: Optional[List[BaseMessage]] = field(default=None)
    research_topic: Optional[str] = field(default=None)

@dataclass(kw_only=True)
class OutputState:
    """The output state for the graph."""
    messages: List[BaseMessage] = field(default_factory=list)
    running_summary: Optional[str] = field(default=None)
    sources_gathered: Optional[List] = field(default=None)