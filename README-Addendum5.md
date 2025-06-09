# LLM Learning NOTES part 5
```cmd
venv\Scripts\activate
jupyter lab
```

## Week 8 **GAME ON**

### What you can now do
* Generate text and code with Frontier Models and Open Source models using APIs and HuggingFace, including tools, assistants and RAG
* Follow a 5 step strategy to solve problems, including dataset curation, making a baseline model, and fine-tuning a Frontier model
* Confidently carry out the end-to-end process for selecting and training specialized open source models that can outperform the Frontier

---

### Week 8 Day 1
#### Learning Objectives
* Use Modal, the serverless platform for AI, to run code remotely
* Deploy an LLM behind an API in the cloud
* Create an Agent that will be incorporated into an autonomous Agentic AI solution

```code
modal setup
    The web browser should have opened for you to authenticate and get an API token.
    If it didn't, please copy this URL into your web browser manually:

    https://modal.com/token-flow/tf-Uf2WoGWXSSfgQooe1bbBPl

    ⠏ Waiting for token flow to complete...
    Web authentication finished successfully!
    Token is connected to the samfire5200 workspace.
    Verifying token against https://api.modal.com
    Token verified successfully!
    Token written to C:\Users\samfi/.modal.toml in profile samfire5200.      

```

.env
``` 
PUSHOVER_USER=uvuq9thwa1nq...
PUSHOVER_TOKEN=aeuhfdmy8293...
```
## We need to set your HuggingFace Token as a secret in Modal

1. Go to modal.com, sign in and go to your dashboard
2. Click on Secrets in the nav bar
3. Create new secret, click on Hugging Face, this new secret needs to be called **hf-secret** because that's how we refer to it in the code
4. Fill in your HF_TOKEN where it prompts you

#### Capstone Project - "The Price is Right"
We will build an Autonomous Agentic AI framework
- Watches for deals being published online
- Estimates the price of the product
- Sends a push notification when it finds great opportunities

SEVEN Agents will collaborate in our framework
- GPT-4o model will identify the deals from an RSS feed
- A frontier-busting fine-tuned model will estimate prices
- A Frontier Model with a massive RAG DB will also be used

#### ✅ **課程目標**

1. **建立一個自治的 Agentic AI 系統**

   * 能自動從網路上找出商品優惠
   * 估算商品價格
   * 發送即時推播通知給使用者

2. **學會代理架構（Agent Framework）設計**

   * 設計多代理協作系統（共七個 agent）
   * 理解各代理功能與協同方式（掃描、估價、通知、規劃等）

3. **部署與應用前沿模型（Frontier Models）**

   * 使用 GPT-4、Fine-tuned LLM、RAG 檢索系統
   * 整合大規模商品資料庫以提升估價準確性

4. **學習工程實務與最佳實踐**

   * 使用 type hints、logging、註解等生產級寫法
   * 引導學生了解部署平台 Modal 的實務操作
   * 鼓勵單元測試與版本控制習慣（Git）

#### 🤖 智能代理系統架構（Agent Architecture）簡介
這個系統是一個多代理協作平台，由以下模組組成：

- 🖥️ The UI（使用者介面）
    使用 Gradio 製作
        - 顯示優惠、估價結果、推薦紀錄

- 🧠 The Agent Framework（代理框架）
    提供記憶與日誌（Memory & Logging）功能
    - 每個 Agent 在此環境中運行、溝通、記錄歷史行為

- 🗂️ Planning Agent（規劃代理）
    負責統籌其他代理的任務執行
    - 決定流程與順序（如：先抓取，再估價，再通知）

- 🔍 Scanner Agent（掃描代理）
    從網路 / RSS feed 中識別值得關注的優惠
    - 是觸發流程的第一步

- 🔮 Ensemble Agent（整合估價代理）
    使用**多個模型**來估算商品價格（如 GPT-4、Fine-tuned 模型、RAG 模型）
    - 提供更準確的價格建議

- 📣 Messaging Agent（推播代理）
    負責發送推播通知
    - 當有好價格出現時，即時通知使用者

#### "As we head towards productionizing" 當邁向產品化階段

邁向**生產環境（Production**時應該注意的**三+1個**良好實作（Best Practices）：

- ✅ Type hints（型別提示）
    為函式的參數與回傳值標註型別
    - 增加程式碼可讀性與 IDE 支援性
    - 有助於靜態分析與錯誤預防

- 💬 Comments（註解）
    在函式或模組中加入說明文字
    - 幫助團隊理解邏輯與意圖
    - 對維護與交接非常關鍵

- 📋 Logging（日誌紀錄）
    將系統活動、錯誤、狀態記錄下來
    - 便於除錯與系統監控
    - 是生產系統必備的監控機制

- Unit Tests（單元測試）***
    - 確保每個模組的功能正確
    - 減少未來修改時引入的錯誤

#### ## 🔍 What is Modal?
- https://modal.com/

**Modal** is a **serverless compute platform** that allows you to:

* Run Python code in the cloud
* Scale effortlessly (concurrent executions)
* Pay only for what you use
* Skip managing VMs, Kubernetes, or Docker directly

http://localhost:8888/lab/tree/week8/day1.ipynb
- http://localhost:8888/lab/tree/week8/hello.py
- http://localhost:8888/lab/tree/week8/llama.py
- http://localhost:8888/lab/tree/week8/pricer_ephemeral.py
- http://localhost:8888/lab/tree/week8/pricer_service.py
    - View Deployment: https://modal.com/apps/samfire5200/main/deployed/pricer-service
- http://localhost:8888/lab/tree/week8/pricer_service2.py
    - View Deployment: https://modal.com/apps/samfire5200/main/deployed/pricer-service

- Using the Modal pricer-service
http://localhost:8888/lab/tree/week8/agents/specialist_agent.py
    - **Pricer** 
        - __init__ => modal.Cls.from_name("pricer-service", "Pricer"); self.pricer = Pricer()
        - **price** => self.pricer.price.remote(description)

#### 💡「What you can now do（你現在可以做到的事）」的中文翻譯如下：

---

* ✅ 使用 Frontier 模型與開源模型來生成文字與程式碼，
  包括透過 API 和 Hugging Face，並能整合工具、助理與 RAG（檢索增強生成）

* ✅ 掌握五步驟策略解決問題，
  包含資料集整理、建立基準模型，以及微調 Frontier 模型

* ✅ 能自信地執行從頭到尾的流程，
  選擇並訓練可超越 Frontier 模型的專用開源模型

* ✅ 使用 **Modal** 平台將大型語言模型（LLMs）部署到生產環境中

---

### Week 8 Day 2
#### Learning Objectives
📅 **今天結束前，你將能夠完成（By end of today）**

* **獨立構建高階 RAG 解法**，不依賴 LangChain
  （🔍 表示你能用輕量方式打造自訂的 RAG）

* **建立專業等級的 Ensemble 模型**
  （🔗 整合多模型提升預測品質）

* **撰寫具備生產等級的程式碼**，可調用多個模型進行推論
  （⚙️ 編寫乾淨、可部署的 API 或服務）

---

建立一個**專業等級的 Ensemble 模型**，指的是透過**結合多個模型的預測結果**，來達到比單一模型更準確、更穩定的預測效果。這是機器學習中常見的高階技巧，廣泛應用於分類、回歸、生成任務、甚至大型語言模型的推理整合。

---

## ✅ 什麼是 Ensemble 模型？

**Ensemble（集成）** = **多個模型的智慧加總**

它不是單靠一個模型做決策，而是**同時運行多個模型**，然後整合他們的輸出，例如：

* 多個模型預測價格 → **平均/加權/投票** → 統一價格結果
* 多個 GPT 模型回答 → 對齊一致的回應

---

## 🧱 專業級 Ensemble 的構成方式

### 🎯 典型方法如下：

| 方法                  | 說明                    | 適合場景       |
| ------------------- | --------------------- | ---------- |
| **Voting（投票法）**     | 用於分類任務，多數票決定最終結果      | 情感分析、文本分類  |
| **Averaging（平均法）**  | 多模型輸出的數值做平均           | 價格預測、數值回歸  |
| **Stacking（堆疊法）**   | 用一個新的模型來學習如何組合多個模型的輸出 | 高階分類/預測系統  |
| **Weighted Voting** | 不同模型依照表現給不同權重         | 多模型融合，重視精度 |

---

## 🧠 在 LLM 中的 Ensemble 應用（例如你的專案）

專案（定價系統）中，使用了 **Ensemble 概念**：

> 整合 GPT-4、Fine-tuned LLaMA、RAG+Chroma 三種模型，來估算商品價格 → 交由 **Ensemble Agent** 做決策

這就是**多模型預測 + 策略融合 + 選擇最佳輸出**，非常符合專業 Ensemble 模型的設計。

---

## 🚀 專業等級 Ensemble 要素

1. **異質模型**：不同架構/訓練背景的模型（如 GPT + LLaMA + DistilBERT）
2. **錯誤互補性**：一個模型錯時，其他能補強
3. **融合策略明確**：平均？投票？還是訓練一個 Meta-Model？
4. **效能考量**：集成不應過度拖慢速度或佔用資源

---

## ✅ 小結

> 建立專業等級的 Ensemble 模型 =
> **結合多模型優勢** + **選擇合適融合策略** + **達成更準確穩定的輸出**

這是在 AI 工程實作中邁向**產品等級、商業應用**的重要步驟。

---

## 🎓 Capstone Project - "The Price Is Right"

### 🧠 我們將建立一個自主的 Agentic AI 架構（Autonomous Agentic AI framework）

* 🔍 持續監控網路上發布的新優惠
* 💰 預估商品的價格
* 📲 當發現絕佳機會時發送**推播**通知

---

### 🤖 我們的架構中將有 **七個智能代理（SEVEN Agents）** 協作：

* 🧠 GPT-4o 模型將從 RSS feed 中識別優惠 (Scanner Agent)
* 🛠️ 我們的 frontier-busting 微調模型將進行價格預估 (Ensemble Agent)
    - 整合多個模型的價格預測結果，Fine-tuned LLM + RAG 模型
* 🗂️ 我們將使用一個搭配龐大 RAG 資料庫的 Frontier 模型 
    - 搭配 Chroma 等向量資料庫，進行語境強化價格預估，結果輸出給 Ensemble Agent 作融合。

---

## 🤖 Agent Workflows（代理工作流程）

### 1️⃣ 使用介面與平台

* **The UI**
  📍 使用 Gradio 建立的前端介面
  **中文：使用者介面（Gradio）**

* **The Agent Framework**
  📍 管理記憶與日誌紀錄 (deal_agent_framework.py)
  **中文：代理框架（具備記憶與日誌功能）**

* **Planning Agent**1
  📍 負責統籌各代理行為 (planning_agent.py)
  **中文：規劃代理（負責協調任務流程）**

---

### 2️⃣ 處理與預測代理群

* **Scanner Agent**2
  📍 識別潛在優惠（例如從 RSS Feed 擷取, scanner_agent.py）
  **中文：掃描代理（發現潛力商品）**

* **Ensemble Agent**3
  📍 整合多個模型的估價結果 (ensemble_agent.py)
  **中文：集成代理（彙總多模型價格預測）**

    * **Frontier Agent**4
    📍 使用 RAG 架構進行估價 (frontier_agent.py - DeepSeek)
    **中文：前沿代理（使用 RAG 估價）**

    * **Specialist Agent**5
    📍 專門估價的子代理之一 (specialist_agent.py - modal)
    **中文：專家代理（專注於特定估價邏輯）**

    * **Random Forest Agent**6
    📍 傳統機器學習模型（隨機森林）估價 (random_forest_agent.py - sklearn)
    **中文：隨機森林代理（傳統 ML 預測器）**

---

### 3️⃣ 通知系統

* **Messaging Agent**7
  📍 發送推播通知給使用者 (messaging_agent.py)
  **中文：通知代理（傳遞結果給用戶）**

---

### 📌 小結：這個架構的流程順序如下：

```
UI → Agent Framework → Planning Agent → 
  ↳ Scanner Agent → 
      Ensemble Agent ↔ Frontier / Specialist / RF Agents →
          Messaging Agent
```

這是一個**多智能代理協作架構**，將多種模型融合為高品質估價系統，並由 UI 觸發、推播通知結果。

---

http://localhost:8888/lab/tree/week8/day2.0.ipynb 
- Create a RAG vector database with our 100,000 training data
http://localhost:8888/lab/tree/week8/day2.1.ipynb 
- Visualize in 2D (30,000 data points) training data
http://localhost:8888/lab/tree/week8/day2.2.ipynb 
- Visualize in 3D (20,000 data points) training data
http://localhost:8888/lab/tree/week8/day2.3.ipynb 
- Build and test a RAG pipeline with GPT-4o-mini and DeepSeek test data
  - http://localhost:8888/lab/tree/week8/agents/frontier_agent.py 
    - **DeepSeek** or **GPT-4o-mini** for Frontier Agent
  - http://localhost:8888/lab/tree/week8/agents/specialist_agent.py 
    - **Modal** for Specialist Agent (**day1**.ipynb)
http://localhost:8888/lab/tree/week8/day2.4.ipynb 
- Build and test a RAG pipeline with Random Forest (sklearn) test data
  - Saved model: week8/**random_forest_model.pkl**
- Finishing off with Random Forests & Ensemble
  - Saved model: week8/**ensemble_model.pkl** with coefficients for each model
    - Specialist: 0.34
    - Frontier: -0.32
    - RandomForest: -0.13

✅ 如果你用的是 FAISS、Weaviate、Pinecone...
| 向量庫          | 獲取紀錄筆數的方法                                  |
| ------------ | -------------------------------------------------------------- |
| **Chroma**   | `vectorstore._collection.count()`                              |
| **FAISS**    | `len(vectorstore.index)`                                       |
| **Weaviate** | 使用 `client.query.aggregate().with_meta_count()`                |
| **Pinecone** | `index.describe_index_stats()` 回傳 dict 中的 `total_vector_count` |

### Week 8 Day 3
#### Learning Objectives
- Use **Structured Outputs** to ensure frontier models respond with a spec
- Develop even further experience using models to solve problems

#### NEW SKILL - Structured Outputs

- A new ability for Frontier Models
- An alternative to JSON generation
- Specify precisely how you want the model to reply
- Define it with a Class
- The model will create an instance of this Class
- Useful for generating data in precisely a given format; sometimes tools / function-calling will be the better method

http://localhost:8888/lab/tree/week8/day3.ipynb 
  - http://localhost:8888/lab/tree/week8/agents/deals.py
  - http://localhost:8888/lab/tree/week8/agents/scanner_agent.py
  - py test_deals.py
    - This will run and print out the deals found
  - py test_scanner_agent.py
    - This will run the Scanner Agent (using gpt-4o-mini) and print out the deals found

---

### 🔄 **Main Difference in Return Values**

| Aspect                  | `ScrapedDeal.fetch()` in `deals.py`          | `ScannerAgent.scan()` in `scanner_agent.py`                   |
| ----------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| **Raw vs. Refined**     | Returns **raw scraped deals** from RSS feeds | Returns **refined top 5 deals**, chosen via LLM and filtered  |
| **Uses AI Filtering**   | ❌ No                                         | ✅ Yes — OpenAI selects top deals with good details and price  |
| **Output Type**         | `List[ScrapedDeal]`                          | `Optional[DealSelection]`                                     |
| **Duplicate Filtering** | ❌ No                                         | ✅ Yes — filters out deals already in memory                   |
| **Deal Quantity**       | Dozens or more per feed                      | Exactly 5 selected deals                                      |
| **User Intent**         | For loading raw data                         | For getting high-quality deal suggestions to show or act upon |

---

### Week 8 Day 4
#### Learning Objectives
Here is the text from the image:
* Define Agent Frameworks, Agentic Workflows in more detail
* Build an Agent Framework that sends push notifications with great deals

#### The Hallmarks of an Agentic AI solution

- Breaking a larger problem into smaller steps carried out by individual processes / models
- Using Tools / Function Calling / Structured Outputs
- An Agent Environment in which Agents can collaborate
- A Planning Agent that coordinates activities
- Autonomy & Memory – existing beyond a chat with a human

#### Example Agentic AI solution
是一個**Agentic AI 解決方案的真實案例** 所列的五個特徵：

---

### 🎯 **真實案例：AI 智慧客服系統（例如：電商平台的智慧客服）**

---

#### ✅ **1. 拆解大型問題為小任務（Breaking a larger problem into smaller steps）**

當使用者詢問「我想查詢上週的訂單狀態並取消其中一筆」，這個複雜請求會被拆成：

* 查找使用者的訂單
* 篩選出上週的訂單
* 顯示訂單狀態
* 接收並執行取消請求

每個步驟由不同的模組處理，確保準確性與可擴展性。

---

#### 🛠 **2. 使用工具/函式呼叫/結構化輸出（Using Tools / Function Calling / Structured Outputs）**

AI Agent 不直接回答所有問題，而是：

* 調用「訂單查詢 API」獲取資訊
* 使用「取消訂單函式」處理指令
* 輸出結構化 JSON 回應給前端，如：

  ```json
  { "order_id": "A123", "status": "cancelled" }
  ```

這樣有助於與系統整合，避免自由文字產生錯誤。

---

#### 🤝 **3. 多 Agent 協作環境（An Agent Environment in which Agents can collaborate）**

此系統可能有多個 Agent：

* **客服語言 Agent**：理解並分析語意
* **訂單管理 Agent**：處理查詢與修改
* **推薦商品 Agent**：根據取消原因推薦替代產品

這些 Agent 在一個環境中協同合作，彼此溝通完成任務。

---

#### 🧠 **4. 有計畫的協調者（A Planning Agent that coordinates activities）**

一個中央「規劃 Agent」根據對話上下文來決定：

* 哪個子 Agent 需要被呼叫
* 執行順序為何
* 如何合併結果給使用者回應

這就是像 Workflow Engine 或 LangGraph 的應用。

---

#### 🧠 **5. 擁有記憶與持續性（Autonomy & Memory - existing beyond a chat）**

系統會記住：

* 使用者的偏好（如付款方式、常購商品）
* 過往的查詢紀錄與處理流程
* 未完成的任務（如退款審核中）

即使使用者隔天回來，也能接續處理。

---

這是一個**典型的 Agentic AI 應用實例**，結合多個模組、工具與 Agent，實現自動化、智能化的客服體驗。

#### Messaging Agent with Pushover Notifications
- https://pushover.net/
  - PUSHOVER_USER=uvuq9thwa...
  - PUSHOVER_TOKEN=aeuhfdmy82...

- Install Pushover Notifications App on your mobile
```bash
curl -s \
  --form-string "token=aeuhfdmy82..." \
  --form-string "user=uvuq9thwa1nqtr..." \
  --form-string "message=� 測試訊息 from llm_engineering" \
  --form-string "title=llm_engineering" \
  https://api.pushover.net/1/messages.json
```

#### Message Agent & Planning Agent
http://localhost:8888/lab/tree/week8/day4.ipynb
  - Messaging Agent (Pushover) => http://localhost:8888/lab/tree/week8/agents/messaging_agent.py
  - Planning Agent => http://localhost:8888/lab/tree/week8/agents/planning_agent.py
    - py test_planner.py

```cmd
cd week8
py deal_agent_framework.py
  ...
  Planning Agent has completed and returned: deal=Deal(product_description=...)

py price_is_right.py

```

#### 🔁 Agent Workflows（代理人工作流程）

### 1. **The UI** （介面）

* **In Gradio**
* 提供使用者與系統互動的前端介面。

### 2. **The Agent Framework** (deal_agent_framework.py)

* **Memory, logging**
* 管理系統記憶與日誌紀錄。 (memory.json)

### 3. **Planning Agent** (planning_agent.py)

* **Coordinates activities**
* 負責協調整個流程中各代理人的執行順序與依賴關係。

---

### 4. **Scanner Agent** (scanner_agent.py)

* **Identifies promising deals**
* 使用 AI 篩選出具有潛力的商品優惠。

---

### 5. **Ensemble Agent** (ensemble_agent.py)

* **Estimates prices**
* 使用集成模型估算每個商品的價格。

---

### 6. **Messaging Agent** (messaging_agent.py)

* **Sends push notifications**
* 傳送通知，例如推播或電郵。

---

### 7. **Frontier Agent** (frontier_agent.py)

* **RAG pricer**
* 使用 RAG（Retrieval-Augmented Generation）估價器進行價格推理。

---

### 8. **Specialist Agent** (specialist_agent.py)

* **Estimates prices**
* 專家模型，可能針對特定類別商品進行更細緻的價格評估。

---

### 9. **Random Forest Agent** (random_forest_agent.py)

* **Estimates prices**
* 使用隨機森林模型作為其中一種價格預測方法。

---

🧠 中文說明：各代理人功能與流程
| 步驟 | 代理人名稱                   | 功能說明                                            |
| -- | ----------------------- | ----------------------------------------------- |
| ①  | **UI**（介面）              | 使用者透過 Gradio 介面啟動流程                             |
| ②  | **Agent Framework**     | 初始化記憶體、日誌系統與向量儲存                                |
| ③  | **Planning Agent**      | 負責協調執行各代理人順序與任務                                 |
| ④  | **Scanner Agent**       | 從 RSS feeds 中使用 GPT 模型篩選出「具描述性 + 有價格」的 top 5 商品 |
| ⑤  | **Ensemble Agent**      | 負責呼叫多個價格模型代理人，合併不同模型的估價結果                       |
| ⑥  | **Frontier Agent**      | 使用 RAG 模型（檢索增強生成）對商品進行語意推論價格                    |
| ⑦  | **Specialist Agent**    | 對特定類別（如電子、玩具）進行更精細估價                            |
| ⑧  | **Random Forest Agent** | 應用 ML 的 RF 模型作為傳統數值預測方法                         |
| ⑨  | **Messaging Agent**     | 將最終結果通知使用者，發送 email、push 等通知                    |

### Week 8 Day 5
#### Learning Objectives
- Master AI and LLM Engineering
- Build an Agentic AI solution that can be deployed in production by Gradio and Modal

http://localhost:8888/lab/tree/week8/day5.ipynb
  - py price_is_right_final.py
    - timer = gr.Timer(value=300, active=True) # every 5 minutes

```code
[2025-06-09 18:19:49 +0800] Kicking off Planning Agent
[2025-06-09 18:19:49 +0800] [Planning Agent] Planning Agent is kicking off a run
[2025-06-09 18:19:49 +0800] [Scanner Agent] Scanner Agent is about to fetch deals from RSS feed
[2025-06-09 18:20:13 +0800] [Scanner Agent] Scanner Agent received 82 deals not already scraped
[2025-06-09 18:20:13 +0800] [Scanner Agent] Number of deals fetched 82
[2025-06-09 18:20:13 +0800] [Scanner Agent] Scanner Agent is calling OpenAI using Structured Output
[2025-06-09 18:20:25 +0800] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
[2025-06-09 18:20:25 +0800] [Scanner Agent] Scanner Agent received 5 selected deals with price>0 from OpenAI
[2025-06-09 18:20:25 +0800] [Planning Agent] Planning Agent is pricing up a potential deal
[2025-06-09 18:20:25 +0800] [Ensemble Agent] Running Ensemble Agent - collaborating with specialist, frontier and random forest agents
[2025-06-09 18:20:25 +0800] [Specialist Agent] Specialist Agent is calling remote fine-tuned model
...
[2025-06-09 18:21:29 +0800] [Ensemble Agent] Ensemble Agent complete - returning $343.81
[2025-06-09 18:21:29 +0800] [Planning Agent] Planning Agent has processed a deal with discount $63.81
[2025-06-09 18:21:29 +0800] [Planning Agent] Planning Agent has identified the best deal has discount $169.10
[2025-06-09 18:21:29 +0800] [Messaging Agent] Messaging Agent is sending a push notification
[2025-06-09 18:21:30 +0800] [Messaging Agent] Messaging Agent has completed
[2025-06-09 18:21:30 +0800] [Planning Agent] Planning Agent has completed a run
[2025-06-09 18:21:30 +0800] Planning Agent has completed and returned: deal=Deal(product_description='The HP Victus laptop features the powerful 12th-Gen Intel Core i5 processor paired with an NVIDIA GeForce RTX 3050 GPU, making it an excellent choice for gamers and content creators. This laptop boasts an impressive 15.6" Full HD display, ensuring vibrant visuals, and comes with 8GB of RAM and a 512GB SSD for swift multitasking and storage. Priced at $510 with free shipping, it offers performance and portability for on-the-go users.', price=510.0, url='https://www.dealnews.com/HP-Victus-12-th-Gen-i5-15-6-Laptop-w-NVIDIA-Ge-Force-RTX-3050-for-510-free-shipping/21742690.html?iref=rss-c39') estimate=679.0962172666656 discount=169.09621726666558

```

### Agentic AI Project Q&A
#### Convert md to word
- cd week5\community-contributions
- py convert_md_to_word.py

#### Agentic AI project Q&A
- http://localhost:8888/lab/tree/week5/community-contributions/day5.markdown_llm_kw_word.ipynb
  - What is Embeddings?
  - What are the steps to create a dataset?
  - What is RAG?
  - What is Agentic AI?