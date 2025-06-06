import asyncio
from agents.deals import ScrapedDeal
from agents.deals_async import ScrapedDealAsync

def test_sync():
    print("🔍 正在執行同步抓取...")
    deals = ScrapedDeal.fetch(show_progress=True)
    print(f"✅ 同步取得 {len(deals)} 筆資料")

async def test_async():
    print("🔍 正在執行非同步抓取...")
    async_deals = await ScrapedDealAsync.fetch_async(show_progress=True)
    print(f"✅ 非同步取得 {len(async_deals)} 筆資料")

if __name__ == "__main__":
    # test_sync()  # 若要測試同步可取消這行註解
    asyncio.run(test_async())  # ✅ 正確執行 async function
