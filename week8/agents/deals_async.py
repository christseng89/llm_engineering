# Asynchronous Deal Fetching
import asyncio
import feedparser
from bs4 import BeautifulSoup
from typing import List, Dict
from typing_extensions import Self
import re
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from agents.deals_common import Deal

import nest_asyncio  # ✅ 加這行
nest_asyncio.apply()  # ✅ 加這行

# ✅ 定義 RSS feeds 來源
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

# ✅ 非同步處理類別
class ScrapedDealAsync:
    """
    An Async class to represent a Deal retrieved from an RSS feed
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, title: str, summary: str, url: str, details: str, features: str):
        self.title = title
        self.summary = summary
        self.url = url
        self.details = details
        self.features = features

        # Try to extract a price
        match = re.search(r"\$(\d+[\.\d+]*)", self.title)
        if match:
            try:
                self.price = float(match.group(1))
            except ValueError:
                self.price = 0.0  # ⛔ 無法轉換為 float 則設為 0
        else:
            self.price = 0.0


    def to_deal(self) -> Deal:
        """
        Return a Deal object from the scraped info
        """
        return Deal(product_description=self.summary, price=self.price, url=self.url)

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    async def _fetch_html(cls, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    @classmethod
    async def from_entry(cls, entry: Dict[str, str]) -> Self:
        title = entry['title']
        summary = cls.extract(entry['summary'])
        url = entry['links'][0]['href']

        html = await cls._fetch_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        content_div = soup.find('div', class_='content-section')
        content = content_div.get_text() if content_div else ""
        content = content.replace('\nmore', '').replace('\n', ' ')
        if "Features" in content:
            details, features = content.split("Features", 1)
        else:
            details = content
            features = ""
        return cls(title, summary, url, details, features)

    @classmethod
    async def _fetch_feed(cls, feed_url: str) -> List[Self]:
        deals = []
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:10]
        for entry in entries:
            deal = await cls.from_entry(entry)
            deals.append(deal)
            await asyncio.sleep(0.5)
        return deals

    @classmethod
    async def fetch_async(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds using asyncio
        """
        tasks = [cls._fetch_feed(url) for url in feeds]
        if show_progress:
            all_results = await tqdm_asyncio.gather(*tasks, desc="📦 Fetching All Feeds")
        else:
            all_results = await asyncio.gather(*tasks)
        return [deal for sublist in all_results for deal in sublist]

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Wrapper to allow synchronous calling environment to still use async code
        """
        return cls.fetch_async(show_progress=show_progress)
        # try:
        #     loop = asyncio.get_event_loop()
        # except RuntimeError:
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)

        # if loop.is_running():
        #     # ✅ 如果事件迴圈已經在跑，使用 ensure_future 搭配 nest_asyncio
        #     future = asyncio.ensure_future(cls.fetch_async(show_progress=show_progress))
        #     return loop.run_until_complete(future)
        # else:
        #     return loop.run_until_complete(cls.fetch_async(show_progress=show_progress))

    @staticmethod
    def extract(summary: str) -> str:
        """
        Clean and extract key parts from the summary
        """
        return summary.replace('\n', '').strip()
