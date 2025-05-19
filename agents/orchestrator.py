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
        # Get stock market data
        stock_data = await self.stock_agent.process(input_data)
        
        # Get news articles
        news_data = await self.web_search_agent.process(input_data)
        
        # Prepare data for LLM processing
        llm_input = {
            'symbol': input_data['symbol'],
            'articles': [article['content'] for article in news_data]
        }
        
        # Generate summary using LLM
        summary = await self.llm_agent.process(llm_input)
        
        # Combine all results
        return {
            'stock_data': stock_data,
            'news_articles': news_data,
            'summary': summary,
            'timestamp': input_data.get('timestamp', None)
        }
    
    async def cleanup(self) -> None:
        """Clean up all sub-agents."""
        for agent in self.agents:
            await agent.cleanup()
        await super().cleanup() 