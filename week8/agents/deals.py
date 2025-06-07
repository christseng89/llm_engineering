# Synchronous Deal Fetching
from typing import List, Dict #, Self # 適用於 Python < 3.11
from bs4 import BeautifulSoup
import feedparser
from tqdm import tqdm
import requests
import time
from typing_extensions import Self  # 適用於 Python < 3.11
import re
import os

# ✅ Load .env for environment variables like FETCH_ASYNC
from dotenv import load_dotenv
load_dotenv()

# For async support
import asyncio
import aiohttp
import sys
import platform

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

class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry['title']
        self.summary = extract(entry['summary'])
        self.url = entry['links'][0]['href']
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, 'html.parser')
        content_div = soup.find('div', class_='content-section')
        if content_div:
            content = content_div.get_text()
            content = content.replace('\nmore', '').replace('\n', ' ')
            if "Features" in content:
                self.details, self.features = content.split("Features", 1)
            else:
                self.details = content
                self.features = ""
        else:
            self.details = "No content-section found"
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

    @classmethod
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds
        """
        fetch_async = os.environ.get("FETCH_ASYNC", "False").lower() == "true"
        if fetch_async:
            # ✅ Fix: Windows workaround to avoid RuntimeError from asyncio.run()
            if platform.system() == 'Windows' and sys.version_info >= (3, 8):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(cls._fetch_async(show_progress))
            else:
                return asyncio.run(cls._fetch_async(show_progress))
        else:
            deals = []
            feed_iter = tqdm(feeds) if show_progress else feeds
            for feed_url in feed_iter:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    deals.append(cls(entry))
                    time.sleep(0.5)
            return deals

    @classmethod
    async def _fetch_async(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve deals using concurrent fetching for faster performance
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds

        async with aiohttp.ClientSession() as session:
            tasks = []
            for feed_url in feed_iter:
                tasks.append(cls._fetch_entries_from_feed(feed_url, session))
            all_entries = await asyncio.gather(*tasks)

            deal_tasks = []
            for entries in all_entries:
                for entry in entries[:10]:
                    deal_tasks.append(cls.from_entry_async(entry, session))
            deals = await asyncio.gather(*deal_tasks)
        return deals

    @staticmethod
    async def _fetch_entries_from_feed(feed_url: str, session: aiohttp.ClientSession) -> List[Dict[str, str]]:
        """
        Fetch and parse entries from one RSS feed URL
        """
        async with session.get(feed_url) as response:
            content = await response.read()
            feed = feedparser.parse(content)
            return feed.entries

    @classmethod
    async def from_entry_async(cls, entry: Dict[str, str], session: aiohttp.ClientSession) -> Self:
        """
        Asynchronously create a ScrapedDeal object from a feed entry
        """
        self = cls.__new__(cls)
        self.title = entry['title']
        self.summary = extract(entry['summary'])
        self.url = entry['links'][0]['href']

        try:
            async with session.get(self.url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                content_div = soup.find('div', class_='content-section')
                if content_div:
                    content = content_div.get_text()
                    content = content.replace('\nmore', '').replace('\n', ' ')
                    if "Features" in content:
                        self.details, self.features = content.split("Features", 1)
                    else:
                        self.details = content
                        self.features = ""
                else:
                    self.details = "No content-section found"
                    self.features = ""
        except Exception as e:
            self.details = "Error loading details"
            self.features = ""
            print(f"⚠️ Error fetching {self.url}: {e}")

        return self
