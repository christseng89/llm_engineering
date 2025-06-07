import os
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from items import Item, save_chunk_to_disk, load_chunk_from_disk  # Do not modify this import

CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49
CACHE_DIR = "cache"

def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_dataset")

def process_chunk_static(name: str, chunk_index: int, chunk) -> List[Item]:
    """
    Static-compatible chunk processor (for multiprocessing)
    Shows messages when loading from disk or generating.
    """
    cached = load_chunk_from_disk(name, chunk_index)
    if cached is not None:
        print(f"📦 Loaded chunk {chunk_index} from disk ({name})", flush=True)
        return cached

    print(f"⚙️ Generating chunk {chunk_index} for {name}...", flush=True)
    results = []
    for dp in chunk:
        try:
            price_str = dp['price']
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(dp, price)
                    if item.include:
                        item.category = name
                        results.append(item)
        except ValueError:
            continue

    save_chunk_to_disk(results, name, chunk_index)
    print(f"💾 Saved chunk {chunk_index} to disk ({name})", flush=True)
    return results

def unpack_and_process_chunk(args: Tuple[str, int, list]) -> List[Item]:
    """
    Helper to unpack arguments for multiprocessing
    """
    return process_chunk_static(*args)

class ItemLoader:

    def __init__(self, name: str, show_progress: bool = True):
        self.name = name
        self.dataset = None
        self.show_progress = show_progress

    def chunk_generator(self) -> List[Tuple[int, list]]:
        """
        Iterate over the Dataset, yielding (chunk_index, chunk)
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            chunk_index = i // CHUNK_SIZE
            chunk = self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))
            yield (chunk_index, chunk)

    def load_in_parallel(self, workers: int) -> List[Item]:
        """
        Use concurrent.futures to process chunks of datapoints with parallelism
        """
        results = []
        chunk_data = [(self.name, idx, chunk) for idx, chunk in self.chunk_generator()]
        chunk_count = len(chunk_data)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            mapped = pool.map(unpack_and_process_chunk, chunk_data)
            if self.show_progress:
                mapped = tqdm(mapped, total=chunk_count)
            for batch in mapped:
                results.extend(batch)
        return results

    def load(self, workers: int = 8) -> List[Item]:
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """
        start = datetime.now()
        print(f"🔍 Loading dataset: {self.name}", flush=True)

        path = cache_path(self.name)
        if os.path.exists(path):
            print(f"📂 Loading from cache: {path}", flush=True)
            self.dataset = load_from_disk(path)
        else:
            print("⬇️ Downloading from Hugging Face Hub...", flush=True)

            # 🛡️ Retry logic for unstable download issues
            for attempt in range(3):
                try:
                    self.dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        f"raw_meta_{self.name}",
                        split="full",
                        trust_remote_code=True
                    )
                    break  # ✅ Download succeeded
                except Exception as e:
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}", flush=True)
                    if attempt < 2:
                        print("🔁 Retrying in 5 seconds...", flush=True)
                        from time import sleep
                        sleep(5)
                    else:
                        raise RuntimeError(f"❌ Failed to download dataset '{self.name}' after 3 attempts.") from e

            print(f"💾 Caching to: {path}", flush=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.dataset.save_to_disk(path)

        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"✅ Completed {self.name} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins", flush=True)
        return results
