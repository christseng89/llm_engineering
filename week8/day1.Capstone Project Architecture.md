# Capstone Project Architecture
Capstone Project（價格合理 The Price is Right）的架構可以簡單分為以下幾個關鍵模組與代理（Agents）：

---

## 🌐 整體架構簡介

### 1️⃣ **使用者介面（UI）**

* 使用 **Gradio** 建立簡單直觀的 Web 前端
* 顯示目前抓到的優惠資訊
* 顯示模型估價結果與推播紀錄

---

### 2️⃣ **代理框架（Agent Framework）**

* 自行設計的 Agent 執行環境，具有：

  * 記憶功能：記錄歷史推薦
  * Logging：方便追蹤行為與除錯
  * 模組化：支援多個代理並行運作

---

### 3️⃣ **核心代理（共七個）**

| Agent 類型                    | 功能說明                                      |
| --------------------------- | ----------------------------------------- |
| 📡 **Scanner Agent**        | 從 RSS Feed 中抓取最新優惠資訊                      |
| 🤖 **Ensemble Agent**       | 整合多個估價模型，提供更準確的價格預測                       |
| 📤 **Messaging Agent**      | 自動推播通知（如手機通知或簡訊）                          |
| 🧠 **3 個 Estimator Agents** | 各自使用不同模型（GPT-4、Fine-tuned LLM、RAG）來預測商品價格 |
| 🗂️ **Planning Agent**      | 控制與協調所有代理之間的流程                            |

---

### 4️⃣ **模型層（LLM Models）**

* **GPT-4 或 GPT-4 Mini**：用來解析 RSS 資料與提取關鍵資訊
* **自有 Fine-tuned 模型**：專門針對估價任務訓練
* **Frontier 模型 + RAG DB**：

  * 搭配 40 萬筆 Amazon 資料建成的大型 Chroma 向量資料庫
  * 用作語境補強，提高價格判斷準確性

---

### 5️⃣ **部署平台：Modal**

* 雲端運行代理與模型
* 支援 Serverless 部署與計費
* 適合 AI 模型快速迭代與實驗

---

## 📌 總結一句話：

這是一個結合多代理、多模型、資料抓取與自動推播的完整 AI 架構平台，具備生產環境潛力且具備模組化、擴展性高的設計。
