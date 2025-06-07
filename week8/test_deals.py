import os
from dotenv import load_dotenv
from agents.deals import ScrapedDeal

# ✅ Load environment variables from `.env` file
load_dotenv()

def test_deals():
    deals = ScrapedDeal.fetch(show_progress=True)
    print(f"✅ Retrieved {len(deals)} deals.")
    for deal in deals[:3]:  # show a few for verification
        print(deal.describe())
        print("=" * 40)

if __name__ == "__main__":
    fetch_async = os.getenv("FETCH_ASYNC", "False").lower() == "true"
    mode = "asynchronous" if fetch_async else "synchronous"
    print(f"🚀 Running in {mode} mode (FETCH_ASYNC={fetch_async})...")
    test_deals()
