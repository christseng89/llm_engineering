import os
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from concurrent.futures import ProcessPoolExecutor
from typing import List
from items import Item  # ✅ Import the full Item class


CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49
CACHE_DIR = "cache"

def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_dataset")

class ItemLoader:

    def __init__(self, name: str):
        self.name = name
        self.dataset = None

    def from_datapoint(self, datapoint) -> Item:
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """        
        try:
            price_str = datapoint['price']
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(datapoint, price)
                    return item if item.include else None
        except ValueError:
            return None

    def from_chunk(self, chunk) -> List[Item]:
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        return [self.from_datapoint(dp) for dp in chunk if self.from_datapoint(dp)]

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """        
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers: int) -> List[Item]:
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        """        
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        for result in results:
            result.category = self.name
        return results

    def load(self, workers: int = 8) -> List[Item]:
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """        
        start = datetime.now()
        print(f"\n🔍 Loading dataset: {self.name}", flush=True)

        path = cache_path(self.name)
        if os.path.exists(path):
            print(f"✅ Loading from cache: {path}", flush=True)
            self.dataset = load_from_disk(path)
        else:
            print("⬇️ Downloading from Hugging Face Hub...", flush=True)
            self.dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_meta_{self.name}",
                split="full",
                trust_remote_code=True
            )
            print(f"💾 Caching to: {path}", flush=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.dataset.save_to_disk(path)

        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"✅ Completed {self.name} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins", flush=True)
        return results
