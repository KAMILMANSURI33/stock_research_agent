import aiohttp
from typing import List, Dict, Any
from datetime import datetime, timedelta
from newspaper import Article
from .base_agent import BaseAgent
from config import NEWS_API_KEY, MAX_NEWS_ARTICLES
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

logger = logging.getLogger(__name__)

class WebSearchAgent(BaseAgent):
    """Agent responsible for searching and gathering news articles."""
    
    def __init__(self):
        super().__init__("WebSearch")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize the aiohttp session."""
        self.session = aiohttp.ClientSession()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _fetch_news_api_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Fetch news from News API with retry logic."""
        if not self.session:
            await self.initialize()
        
        async with self.session.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': f'"{symbol}" stock',
                'from': start_date.isoformat(),
                'to': end_date.isoformat(),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': NEWS_API_KEY
            }
        ) as response:
            if response.status != 200:
                logger.error(f"News API error: {response.status}")
                return []
            data = await response.json()
            articles = data.get('articles', [])
            
            articles.sort(
                key=lambda x: datetime.fromisoformat(x['publishedAt'].replace('Z', '+00:00')),
                reverse=True
            )
            
            return articles[:MAX_NEWS_ARTICLES]
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4))
    async def _fetch_article_content(self, article_url: str) -> str:
        """Fetch and parse article content with retry logic."""
        try:
            news_article = Article(article_url)
            await asyncio.get_event_loop().run_in_executor(None, news_article.download)
            await asyncio.get_event_loop().run_in_executor(None, news_article.parse)
            return news_article.text
        except Exception as e:
            logger.error(f"Error fetching article content from {article_url}: {e}")
            return ""
    
    async def process(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for news articles about the given stock symbol."""
        symbol = input_data['symbol']
        days = input_data.get('days', 7)
        
        logger.info(f"Fetching news for symbol: {symbol} for the last {days} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        articles_data = await self._fetch_news_api_data(symbol, start_date, end_date)
        
        processed_articles = []
        tasks = []
        
        for article in articles_data:
            if not article.get('url'):
                continue
                
            tasks.append(self._fetch_article_content(article['url']))
        
        if tasks:
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            for article, content in zip(articles_data, contents):
                if isinstance(content, Exception):
                    logger.error(f"Error processing article: {content}")
                    continue
                    
                if not content:
                    continue
                
                published_at = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'published_at': published_at.isoformat(),
                    'source': article.get('source', {}).get('name', ''),
                    'content': content
                })
        
        processed_articles.sort(key=lambda x: x['published_at'], reverse=True)
        
        logger.info(f"Successfully processed {len(processed_articles)} articles for {symbol}")
        return processed_articles
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
        await super().cleanup() 