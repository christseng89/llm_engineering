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

### An intensive week lies ahead! After today, you’ll be able to:
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

#### Learning Objectives Capstone Project - "The Price is Right"
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
        - __init__ => modal.Cls.lookup("pricer-service", "Pricer"); self.pricer = Pricer()
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
