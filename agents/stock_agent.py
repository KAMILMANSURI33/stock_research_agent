import yfinance as yf
from typing import Dict, Any
from .base_agent import BaseAgent
from config import ALPHAVANTAGE_API_KEY
import aiohttp
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class StockAgent(BaseAgent):
    """Agent responsible for fetching stock market data."""
    
    def __init__(self):
        super().__init__("Stock")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize the aiohttp session."""
        self.session = aiohttp.ClientSession()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _fetch_alpha_vantage_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage with retry logic."""
        if not self.session:
            await self.initialize()
            
        async with self.session.get(
            'https://www.alphavantage.co/query',
            params={
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': ALPHAVANTAGE_API_KEY
            }
        ) as response:
            if response.status != 200:
                logger.error(f"Alpha Vantage API error: {response.status}")
                return {}
            return await response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_yfinance_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Yahoo Finance with retry logic."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
            }
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return {}

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch stock data and market information."""
        symbol = input_data['symbol']
        logger.info(f"Fetching data for symbol: {symbol}")
        
        # Get stock info from Yahoo Finance
        yf_data = self._fetch_yfinance_data(symbol)
        
        # Get additional market data from Alpha Vantage
        av_data = await self._fetch_alpha_vantage_data(symbol)
        quote = av_data.get('Global Quote', {})
        
        result = {
            'symbol': symbol,
            'name': yf_data.get('name', ''),
            'sector': yf_data.get('sector', ''),
            'industry': yf_data.get('industry', ''),
            'market_cap': yf_data.get('market_cap', 0),
            'pe_ratio': yf_data.get('pe_ratio', 0),
            'dividend_yield': yf_data.get('dividend_yield', 0),
            'price': float(quote.get('05. price', 0) or 0),
            'change_percent': float(quote.get('10. change percent', '0%').strip('%') or 0),
            'volume': int(quote.get('06. volume', 0) or 0)
        }
        
        logger.info(f"Successfully fetched data for {symbol}")
        return result
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
        await super().cleanup() 