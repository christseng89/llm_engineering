from agents.deals import ScrapedDeal
from agents.deals_async import AsyncScrapedDeal

def test_sync():
    print("🔍 正在執行同步抓取...")
    deals = ScrapedDeal.fetch(show_progress=True)
    print(f"✅ 同步取得 {len(deals)} 筆資料")

def test_async():
    print("🔍 正在執行非同步抓取...")
    async_deals = AsyncScrapedDeal.fetch(show_progress=True)  # ✅ 無需 await
    print(f"✅ 非同步取得 {len(async_deals)} 筆資料")

if __name__ == "__main__":
    # ✅ 若要測試同步抓取請取消下行註解
    # test_sync()

    # ✅ 測試非同步抓取（可自動切換快取）
    test_async()
