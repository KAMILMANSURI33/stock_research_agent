import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from .web_search_agent import WebSearchAgent
from .stock_agent import StockAgent
from .llm_agent import LLMAgent

class Orchestrator(BaseAgent):
    """Main agent that orchestrates all sub-agents."""
    
    def __init__(self):
        super().__init__("Orchestrator")
        self.web_search_agent = WebSearchAgent()
        self.stock_agent = StockAgent()
        self.llm_agent = LLMAgent()
        self.agents = [self.web_search_agent, self.stock_agent, self.llm_agent]
    
    async def initialize(self) -> None:
        """Initialize all sub-agents."""
        for agent in self.agents:
            await agent.initialize()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input using all sub-agents and combine results."""
        # Get stock market data and news articles concurrently
        stock_data, news_data = await asyncio.gather(
            self.stock_agent.process(input_data),
            self.web_search_agent.process(input_data)
        )
        
        # Lift the provenance marker out of stock_data so the payload shape
        # stays as callers expect, with the source reported alongside it.
        stock_data_source = (
            stock_data.pop('source', None) if isinstance(stock_data, dict) else None
        )

        # Prepare data for LLM processing
        llm_input = {
            'symbol': input_data['symbol'],
            'articles': [article['content'] for article in news_data]
        }
        
        # Generate summary using LLM. It reports how many of the retrieved
        # articles actually fit the model's context window.
        llm_result = await self.llm_agent.process(llm_input)
        
        # Combine all results
        result = {
            'stock_data': stock_data,
            # Which upstream supplied each half of stock_data. Both fail
            # silently, so an exhausted quota is otherwise invisible.
            'stock_data_source': stock_data_source,
            'news_articles': news_data,
            # Explicit so a caller can tell a grounded summary from an empty
            # one without re-deriving it from the articles list.
            'articles_retrieved': len(news_data),
            'articles_used': llm_result['articles_used'],
            # Which retrieved articles actually reached the prompt. Grounding
            # must be scored against these, not against news_articles.
            'articles_used_indices': llm_result['articles_used_indices'],
            'summary': llm_result['summary'],
            'timestamp': input_data.get('timestamp', None)
        }

        # Opt-in only: the exact prompt excerpts are for evaluation, so a
        # harness need not reimplement truncation and drift from it. Omitted
        # entirely by default, leaving production responses unchanged.
        if input_data.get('include_prompt_context'):
            result['prompt_context'] = llm_result['prompt_context']

        return result
    
    async def cleanup(self) -> None:
        """Clean up all sub-agents."""
        for agent in self.agents:
            await agent.cleanup()
        await super().cleanup() 