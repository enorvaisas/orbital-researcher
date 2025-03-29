# Orbital Researcher

A research agent that uses web search to gather information and provide comprehensive answers to questions.

## Features

- Uses LangGraph for a structured research workflow
- Performs web searches using various search APIs (Tavily, DuckDuckGo, Perplexity, SearXNG)
- Generates search queries based on research topics
- Summarizes search results
- Reflects on findings and generates follow-up questions
- Provides comprehensive answers with citations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/enorvaisas/orbital-researcher.git
cd orbital-researcher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for API keys (optional but recommended):
```bash
# For Tavily search
export TAVILY_API_KEY=your_tavily_api_key

# For Anthropic models
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For OpenAI models
export OPENAI_API_KEY=your_openai_api_key

# For Perplexity
export PERPLEXITY_API_KEY=your_perplexity_api_key

# For SearXNG (optional)
export SEARXNG_URL=your_searxng_instance_url
```

## Usage with LangGraph Studio

1. Start the LangGraph Studio server:
```bash
langgraph dev
```

2. Open http://localhost:3000 in your browser

3. Create a new assistant using the "Research" agent

4. Configure the agent with your preferred settings:
   - System prompt
   - Language model
   - Search API
   - Maximum search results
   - Maximum search iterations
   - Whether to fetch full content

5. Ask your research question and watch the agent work!

## Development

The agent is structured as follows:

- `src/research_agent/graph.py`: Contains the main graph logic
- `src/research_agent/configuration.py`: Configuration classes
- `src/research_agent/state.py`: State management
- `src/research_agent/prompts.py`: Prompt templates
- `src/research_agent/utils.py`: Utility functions

## License

MIT