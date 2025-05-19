from typing import Any, Optional
from llama_cpp import Llama
from .base_agent import BaseAgent
from config import LLAMA_MODEL_PATH, MAX_TOKENS, TEMPERATURE, SUMMARY_PROMPT

class LLMAgent(BaseAgent):
    """Agent responsible for LLM-based text processing using Llama."""
    
    def __init__(self):
        super().__init__("LLM")
        self.model: Optional[Llama] = None
    
    async def initialize(self) -> None:
        """Initialize the Llama model."""
        self.model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=MAX_TOKENS,
            n_threads=4
        )
    
    async def process(self, input_data: dict) -> str:
        """Process input data using the Llama model."""
        if not self.model:
            await self.initialize()
        
        # Prepare the prompt
        symbol = input_data.get('symbol', '')
        articles = input_data.get('articles', [])
        articles_text = "\n\n".join([f"Article {i+1}:\n{article}" for i, article in enumerate(articles)])
        
        prompt = SUMMARY_PROMPT.format(
            symbol=symbol,
            articles=articles_text
        )
        
        # Generate response
        response = self.model.create_completion(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=["###"]
        )
        
        return response['choices'][0]['text'].strip()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
        await super().cleanup() 