"""Utility functions for the research agent."""

import os
import re
import json
from typing import Dict, List, Any, Optional, Union

import requests
from bs4 import BeautifulSoup
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Function to strip thinking tokens, e.g., <thinking>...</thinking>
def strip_thinking_tokens(text: str) -> str:
    """Remove thinking tokens from text"""
    return re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

def get_config_value(value: Any) -> Any:
    """Get a value from config or environment variable"""
    if isinstance(value, str) and value.startswith("$"):
        env_var = value[1:]
        return os.environ.get(env_var)
    return value

def load_chat_model(model_string: str) -> BaseChatModel:
    """Load a chat model based on a model string.
    
    Model string should be in the format "provider/model_name".
    Examples: "anthropic/claude-3-5-sonnet-20240620", "openai/gpt-4o"
    """
    if not model_string or "/" not in model_string:
        # Default to Anthropic's Claude
        return ChatAnthropic(model="claude-3-5-sonnet-20240620")
    
    provider, model_name = model_string.split("/", 1)
    
    if provider.lower() == "anthropic":
        return ChatAnthropic(model=model_name)
    elif provider.lower() == "openai":
        return ChatOpenAI(model=model_name)
    elif provider.lower() == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name)
    else:
        # Default fallback
        return ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Web search functions
def tavily_search(query: str, fetch_full_page: bool = False, max_results: int = 3) -> List[Dict]:
    """Search the web using Tavily API"""
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query, max_results=max_results)
        
        if fetch_full_page:
            for result in results:
                try:
                    if result.get("url"):
                        page_content = fetch_webpage_content(result["url"])
                        if page_content:
                            result["page_content"] = page_content
                except Exception as e:
                    print(f"Error fetching content from {result.get('url')}: {e}")
        
        return results
    except Exception as e:
        print(f"Error in Tavily search: {e}")
        return []

def duckduckgo_search(query: str, max_results: int = 5, fetch_full_page: bool = False) -> List[Dict]:
    """Search the web using DuckDuckGo API"""
    try:
        search = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        results = search.results(query)
        
        formatted_results = []
        for result in results:
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "content": result.get("snippet", "")
            }
            
            if fetch_full_page and formatted_result["url"]:
                try:
                    page_content = fetch_webpage_content(formatted_result["url"])
                    if page_content:
                        formatted_result["page_content"] = page_content
                except Exception as e:
                    print(f"Error fetching content from {formatted_result['url']}: {e}")
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    except Exception as e:
        print(f"Error in DuckDuckGo search: {e}")
        return []

def perplexity_search(query: str, search_iteration: int = 0) -> List[Dict]:
    """Search the web using Perplexity API"""
    try:
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            print("No Perplexity API key found")
            return []
            
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"query": query, "max_results": 5}
        
        response = requests.post(
            "https://api.perplexity.ai/search",
            headers=headers,
            json=params
        )
        
        if response.status_code != 200:
            print(f"Error from Perplexity API: {response.status_code}, {response.text}")
            return []
            
        results = response.json()
        
        formatted_results = []
        for result in results.get("results", []):
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("snippet", "")
            }
            formatted_results.append(formatted_result)
            
        return formatted_results
    except Exception as e:
        print(f"Error in Perplexity search: {e}")
        return []

def searxng_search(query: str, max_results: int = 5, fetch_full_page: bool = False) -> List[Dict]:
    """Search the web using SearXNG API"""
    try:
        searx_url = os.environ.get("SEARXNG_URL", "https://searx.be")
        response = requests.get(
            f"{searx_url}/search",
            params={"q": query, "format": "json", "engines": "google,bing,duckduckgo", "language": "en-US"},
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        if response.status_code != 200:
            print(f"Error from SearXNG API: {response.status_code}, {response.text}")
            return []
            
        results = response.json().get("results", [])
        
        formatted_results = []
        for i, result in enumerate(results):
            if i >= max_results:
                break
                
            formatted_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", "")
            }
            
            if fetch_full_page and formatted_result["url"]:
                try:
                    page_content = fetch_webpage_content(formatted_result["url"])
                    if page_content:
                        formatted_result["page_content"] = page_content
                except Exception as e:
                    print(f"Error fetching content from {formatted_result['url']}: {e}")
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    except Exception as e:
        print(f"Error in SearXNG search: {e}")
        return []

def fetch_webpage_content(url: str, max_length: int = 10000) -> str:
    """Fetch content from a webpage"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove scripts, styles and navigation elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator="\n")
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    except Exception as e:
        print(f"Error fetching webpage content: {e}")
        return ""

def deduplicate_and_format_sources(sources: List[Dict], max_tokens_per_source: int = 1000, fetch_full_page: bool = False) -> str:
    """Deduplicate and format search results for use in prompt"""
    if not sources:
        return "No sources found."
        
    seen_urls = set()
    formatted_sources = []
    
    for i, source in enumerate(sources):
        url = source.get("url")
        if url and url in seen_urls:
            continue
            
        if url:
            seen_urls.add(url)
            
        title = source.get("title", f"Source {i+1}")
        content = source.get("content", "")
        
        # Use full page content if available and requested
        if fetch_full_page and "page_content" in source and source["page_content"]:
            content = source["page_content"]
            
        # Truncate content if too long
        if len(content) > max_tokens_per_source:
            content = content[:max_tokens_per_source] + "..."
            
        formatted_source = f"Source: {title}\nURL: {url}\n{content}\n\n"
        formatted_sources.append(formatted_source)
        
    return "\n".join(formatted_sources)

def format_sources(sources: List[Dict]) -> str:
    """Format source list for citations"""
    if not sources:
        return "No sources found."
        
    formatted_sources = []
    
    for i, source in enumerate(sources):
        url = source.get("url", "")
        title = source.get("title", f"Source {i+1}")
        formatted_source = f"{i+1}. [{title}]({url})"
        formatted_sources.append(formatted_source)
        
    return "\n".join(formatted_sources)