import asyncio
import aiohttp

from typing import List, Dict
from bs4 import BeautifulSoup
import feedparser
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import Self  # 適用於 Python < 3.11
from bs4 import BeautifulSoup
import re

from deals import ScrapedDeal  # ✅ 加上這行

# 共用 RSS feeds
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
    "https://www.dealnews.com/c756/Health-Beauty/?rss=1",
    "https://www.dealnews.com/c182/Office-School-Supplies/?rss=1",
    "https://www.dealnews.com/c178/Movies-Music-Books/?rss=1",
    "https://www.dealnews.com/c186/Gaming-Toys/?rss=1",
]

def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, 'html.parser')
    snippet_div = soup.find('div', class_='snippet summary')
    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, 'html.parser').get_text()
        description = re.sub('<[^<]+?>', '', description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace('\n', ' ')


class AsyncScrapedDeal:
    """
    An Async class to represent a Deal retrieved from an RSS feed
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str], content: str):
        """
        Populate this instance based on the provided dict
        """        
        self.title = entry['title']
        self.summary = extract(entry['summary'])
        self.url = entry['links'][0]['href']
        soup = BeautifulSoup(content, 'html.parser')
        content = soup.find('div', class_='content-section')
        content = content.get_text() if content else ""
        content = content.replace('\nmore', '').replace('\n', ' ')
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""

    def __repr__(self):
        """
        Return a string to describe this deal
        """        
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """        
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

class AsyncScrapedDeal(ScrapedDeal):

    @classmethod
    async def fetch_async(cls, show_progress: bool = False) -> List[Self]:
        """
        🔄 原始 async 實作，用來真正並行抓取資料
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        async with aiohttp.ClientSession() as session:
            tasks = [cls._fetch_feed(session, url) for url in feed_iter]
            results = await asyncio.gather(*tasks)
            for batch in results:
                deals.extend(batch)
        return deals

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        ✅ 對外使用：同步函數，包裝 async 實作，支援直接呼叫
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop = asyncio.get_event_loop()

        return loop.run_until_complete(cls.fetch_async(show_progress=show_progress))    

    @classmethod
    async def _fetch_feed(cls, feed_url: str) -> List[Self]:
        feed = feedparser.parse(feed_url)
        async with aiohttp.ClientSession() as session:
            tasks = [cls._fetch_deal(session, entry) for entry in feed.entries[:10]]
            results = await asyncio.gather(*tasks)
            return [deal for deal in results if deal]

    @classmethod
    async def _fetch_deal(cls, session: aiohttp.ClientSession, entry: Dict[str, str]):
        try:
            # 🛡️ 檢查必要欄位，否則跳過
            if 'title' not in entry or 'summary' not in entry or 'links' not in entry or not entry['links']:
                return None
            async with session.get(entry['links'][0]['href']) as resp:
                content = await resp.text()
                return cls(entry, content)
        except Exception as e:
            print(f"❌ Error fetching deal: {e}")
            return None
