"""Research agent graph logic."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Annotated, TypedDict, Optional, cast
from dataclasses import asdict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tool_resources.convert_to_openai import format_tool_to_openai
from langchain_core.pydantic_v1 import Field
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

from research_agent.configuration import Configuration, SearchAPI
from research_agent.state import State, SearchState
from research_agent.prompts import (
    SYSTEM_PROMPT, 
    QUERY_WRITER_INSTRUCTIONS, 
    SUMMARIZER_INSTRUCTIONS,
    REFLECTION_INSTRUCTIONS,
    get_current_date
)
from research_agent.utils import (
    load_chat_model,
    tavily_search,
    duckduckgo_search,
    perplexity_search,
    searxng_search,
    deduplicate_and_format_sources,
    format_sources,
    strip_thinking_tokens
)

# Define configurable parameters for the graph
class GraphConfig(TypedDict, total=False):
    """Configuration schema for research agent graph."""
    
    system_prompt: Annotated[str, Field(
        description="System prompt for the research agent"
    )]
    model: Annotated[str, Field(
        description="Language model to use (format: provider/model_name)"
    )]
    search_api: Annotated[SearchAPI, Field(
        description="Search API to use for web searches"
    )]
    max_search_results: Annotated[int, Field(
        description="Maximum number of search results to return",
        ge=1,
        le=20
    )]
    max_search_iterations: Annotated[int, Field(
        description="Maximum number of search iterations",
        ge=1,
        le=10
    )]
    fetch_full_content: Annotated[bool, Field(
        description="Whether to fetch full content of search results"
    )]

def create_tools() -> List[Dict[str, Any]]:
    """Create tools for the agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "generate_search_query",
                "description": "Generate a search query based on the research topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Rationale for the search query"
                        }
                    },
                    "required": ["query", "rationale"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_results",
                "description": "Summarize the search results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A comprehensive summary of the information from search results"
                        }
                    },
                    "required": ["summary"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reflect_and_continue",
                "description": "Reflect on current findings and decide whether to continue research",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reflection": {
                            "type": "string",
                            "description": "Reflection on current findings"
                        },
                        "continue_research": {
                            "type": "boolean",
                            "description": "Whether to continue research with follow-up questions"
                        },
                        "follow_up_questions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Follow-up questions to explore"
                        }
                    },
                    "required": ["reflection", "continue_research"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_research",
                "description": "Finalize the research and provide a comprehensive answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "final_answer": {
                            "type": "string",
                            "description": "Final comprehensive answer to the research question"
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Sources used in the research"
                        }
                    },
                    "required": ["final_answer"]
                }
            }
        }
    ]

def search_the_web(state: State) -> State:
    """Execute web search based on generated query."""
    config = state.config
    search_state = state.search_state
    
    # Extract the query from the last message (should be the query generation message)
    last_message = state.messages[-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.additional_kwargs.get("function_call"):
        # Something went wrong, no tool call in the last message
        return state
    
    # Extract the query from the function call
    function_call = last_message.additional_kwargs["function_call"]
    
    if function_call["name"] != "generate_search_query":
        # Wrong function was called
        return state
    
    try:
        arguments = json.loads(function_call["arguments"])
        query = arguments.get("query", "")
        
        if not query:
            # No query was generated
            return state
            
        # Execute search based on configured search API
        search_api = config.search_api
        max_results = config.max_search_results
        fetch_full_content = config.fetch_full_content
        
        if search_api == SearchAPI.TAVILY:
            results = tavily_search(query, fetch_full_content, max_results)
        elif search_api == SearchAPI.DUCKDUCKGO:
            results = duckduckgo_search(query, max_results, fetch_full_content)
        elif search_api == SearchAPI.PERPLEXITY:
            results = perplexity_search(query)
        elif search_api == SearchAPI.SEARXNG:
            results = searxng_search(query, max_results, fetch_full_content)
        else:
            # Default to Tavily
            results = tavily_search(query, fetch_full_content, max_results)
            
        # Update search state
        search_state.search_iterations += 1
        search_state.queries.append(query)
        search_state.current_results = results
        search_state.all_results.extend(results)
        
        # Format sources for the model
        formatted_sources = deduplicate_and_format_sources(
            results, 
            fetch_full_page=fetch_full_content
        )
        
        # Add AI message with search results
        human_message = HumanMessage(
            content=f"Here are the search results for query: '{query}':\n\n{formatted_sources}\n\n"
            f"Please summarize these results using the summarize_results function."
        )
        
        return State(
            messages=state.messages + [human_message],
            config=config,
            research_topic=state.research_topic,
            search_state=search_state
        )
        
    except Exception as e:
        # Log error and return original state
        print(f"Error in search_the_web: {e}")
        
        # Add error message
        human_message = HumanMessage(
            content=f"Error performing search: {str(e)}. Please try a different approach."
        )
        
        return State(
            messages=state.messages + [human_message],
            config=config,
            research_topic=state.research_topic,
            search_state=search_state
        )

def call_model(state: State, config: Configuration) -> State:
    """Call the language model with the current state."""
    try:
        # Load model from configuration
        model = load_chat_model(config.model)
        
        # Get messages from state with system message
        formatted_system_prompt = config.system_prompt.format(
            current_date=get_current_date(),
            topic=state.research_topic
        )
        
        # Create message list with appropriate message types
        messages = state.messages.copy()
        
        # Format tools for the model
        tools = create_tools()
        
        # Call the model with tools
        response = model.invoke(
            messages,
            functions=tools if hasattr(model, "functions") else None,
            tools=tools if not hasattr(model, "functions") else None,
            response_format={"type": "json_object"} if hasattr(model, "response_format") else None,
            system=formatted_system_prompt
        )
        
        # Clean up response if needed
        if hasattr(response, "content") and isinstance(response.content, str):
            response.content = strip_thinking_tokens(response.content)
        
        return State(
            messages=state.messages + [response],
            config=config,
            research_topic=state.research_topic,
            search_state=state.search_state
        )
    except Exception as e:
        print(f"Error in call_model: {e}")
        # Return original state on error
        return state

def should_continue_research(state: State) -> str:
    """Determine if research should continue or end."""
    # Check if we've reached max search iterations
    if state.search_state.search_iterations >= state.config.max_search_iterations:
        return "finalize"
    
    # Check last message for tool call
    last_message = state.messages[-1]
    
    if not isinstance(last_message, AIMessage):
        return "search"  # Default to continuing search
        
    # Check if the last message contains a function call
    function_call = last_message.additional_kwargs.get("function_call")
    
    if not function_call:
        return "search"  # No function call, continue research
        
    function_name = function_call.get("name")
    
    if function_name == "generate_search_query":
        return "search"
    elif function_name == "reflect_and_continue":
        try:
            args = json.loads(function_call["arguments"])
            continue_research = args.get("continue_research", True)
            
            if continue_research:
                return "continue"
            else:
                return "finalize"
        except:
            return "continue"  # Default to continuing on error
    elif function_name == "finalize_research":
        return "end"
    else:
        return "continue"  # Default to continuing

def reflect_on_results(state: State) -> State:
    """Add a message prompting reflection on current findings."""
    config = state.config
    search_state = state.search_state
    
    # Generate a prompt for the model to reflect on the research so far
    human_message = HumanMessage(
        content=f"""
You have conducted {search_state.search_iterations} search iterations on the topic: "{state.research_topic}".
 
{REFLECTION_INSTRUCTIONS}

Please use the reflect_and_continue function to share your reflections and determine if additional research is needed.
"""
    )
    
    return State(
        messages=state.messages + [human_message],
        config=config,
        research_topic=state.research_topic,
        search_state=search_state
    )

def prepare_for_next_query(state: State) -> State:
    """Prepare the state for the next query generation."""
    config = state.config
    search_state = state.search_state
    
    # Extract follow-up questions if available
    last_message = state.messages[-1]
    follow_up_questions = []
    
    if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("function_call"):
        function_call = last_message.additional_kwargs["function_call"]
        if function_call["name"] == "reflect_and_continue":
            try:
                args = json.loads(function_call["arguments"])
                follow_up_questions = args.get("follow_up_questions", [])
            except:
                pass
    
    # Create a prompt for generating the next query
    next_question = ""
    if follow_up_questions:
        next_question = follow_up_questions[0]
    
    human_message = HumanMessage(
        content=f"""
Based on your research so far on "{state.research_topic}", please generate a search query to find more information.
{f'Consider this follow-up question: "{next_question}"' if next_question else ""}

{QUERY_WRITER_INSTRUCTIONS}

Please use the generate_search_query function to provide your query and rationale.
"""
    )
    
    return State(
        messages=state.messages + [human_message],
        config=config,
        research_topic=state.research_topic,
        search_state=search_state
    )

def prepare_for_finalization(state: State) -> State:
    """Prepare the state for finalizing the research."""
    config = state.config
    search_state = state.search_state
    
    # Format all sources for citation
    all_sources = search_state.all_results
    formatted_sources = format_sources(all_sources)
    
    human_message = HumanMessage(
        content=f"""
You have completed your research on the topic: "{state.research_topic}".

Please provide a final comprehensive answer that synthesizes all the information you've gathered.
Remember to be thorough, accurate, and well-structured.

Here are all the sources you can cite:
{formatted_sources}

Please use the finalize_research function to provide your final answer and sources.
"""
    )
    
    return State(
        messages=state.messages + [human_message],
        config=config,
        research_topic=state.research_topic,
        search_state=search_state
    )

def create_research_graph(config: GraphConfig) -> StateGraph:
    """Create the research agent graph."""
    # Convert GraphConfig to Configuration
    configuration = Configuration(
        system_prompt=config.get("system_prompt", SYSTEM_PROMPT),
        model=config.get("model", "anthropic/claude-3-sonnet-20240229"),
        search_api=config.get("search_api", SearchAPI.TAVILY),
        max_search_results=config.get("max_search_results", 5),
        max_search_iterations=config.get("max_search_iterations", 3),
        fetch_full_content=config.get("fetch_full_content", False)
    )
    
    # Initialize the workflow graph
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("call_model", lambda state: call_model(state, configuration))
    workflow.add_node("search", search_the_web)
    workflow.add_node("reflect", reflect_on_results)
    workflow.add_node("next_query", prepare_for_next_query)
    workflow.add_node("finalize", prepare_for_finalization)
    
    # Add edges to the graph
    workflow.add_edge("call_model", should_continue_research)
    workflow.add_edge("search", "call_model")
    workflow.add_edge("reflect", "call_model")
    workflow.add_edge("next_query", "call_model")
    workflow.add_edge("finalize", "call_model")
    
    # Connect the router
    workflow.add_conditional_edges(
        "call_model",
        should_continue_research,
        {
            "search": "search",
            "continue": "reflect",
            "finalize": "finalize",
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("next_query")
    
    return workflow.compile()

if __name__ == "__main__":
    # For testing
    config = {
        "system_prompt": SYSTEM_PROMPT,
        "model": "anthropic/claude-3-sonnet-20240229",
        "search_api": SearchAPI.TAVILY,
        "max_search_results": 5,
        "max_search_iterations": 3,
        "fetch_full_content": False
    }
    
    graph = create_research_graph(config)
    
    # Initialize state
    initial_state = State(
        messages=[],
        config=Configuration(**config),
        research_topic="What are the latest advancements in quantum computing?",
        search_state=SearchState()
    )
    
    # Run the graph
    for state in graph.stream(initial_state):
        last_message = state.messages[-1] if state.messages else None
        if last_message:
            print(f"Node: {state.get('node', 'unknown')}")
            if isinstance(last_message, AIMessage) and last_message.additional_kwargs.get("function_call"):
                function_call = last_message.additional_kwargs["function_call"]
                print(f"Function call: {function_call['name']}")
            elif isinstance(last_message, AIMessage):
                print(f"AI: {last_message.content[:100]}...")
            elif isinstance(last_message, HumanMessage):
                print(f"Human: {last_message.content[:100]}...")