# deals_common.py

from typing import List
from pydantic import BaseModel


class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    """    
    product_description: str
    price: float
    url: str

class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """    
    deals: List[Deal]

class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """    
    deal: Deal
    estimate: float
    discount: float

# ------------------------------------------------------
# ✅ 快取處理邏輯（避免每次都抓取 RSS）Synchronous版本 ONLY
# ------------------------------------------------------
import os
import json
from datetime import datetime, timedelta

# 快取檔案名稱與過期時間（單位：小時）
CACHE_FILE = "deals_cache.json"
EXPIRY_HOURS = 6

def is_cache_fresh() -> bool:
    """
    檢查快取檔案是否存在，且未超過設定的過期時間
    """
    if not os.path.exists(CACHE_FILE):
        return False
    modified_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return datetime.now() - modified_time < timedelta(hours=EXPIRY_HOURS)

def save_deals_to_cache(deals: List[Deal]):
    """
    將 List[Deal] 儲存成 JSON 格式的快取檔案
    """
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([deal.dict() for deal in deals], f, ensure_ascii=False, indent=2)

def load_deals_from_cache() -> List[Deal]:
    """
    從快取檔案中讀取並還原為 List[Deal] 物件
    """
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        return [Deal(**d) for d in raw_data]
