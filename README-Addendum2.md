# LLM Learning NOTES part 2
```cmd
venv\Scripts\activate
jupyter lab
```

### Week 3 Day 1
#### Learning Objectives
- Describe the 'HuggingFace' platform and everything it offers
- Look at Models, Datasets and Spaces
- Use Google CoLab to code on a high spec GPU runtime

#### HuggingFace Platform
The ubiquitous platform for LLM Engineers

- Models
    Over 800,000 Open Source Models of all shapes and sizes
        ✅ LLaMA 3 8B / 70B (by Meta)
        ✅ Mistral-7B (dense) and Mixtral 8x7B (MoE)
        ✅ Falcon 7B / 40B
        ✅ Whisper (by OpenAI) for speech-to-text transcription
        ✅ Coqui TTS, Bark, and XTTS 2 for text-to-speech synthesis        
        ✅ Stable Diffusion (by Stability AI) for image generation
        ✅ BGE & E5 Embedding Models for semantic search, RAG, and retrieval tasks
- Datasets
    A treasure trove of 200,000 datasets
- Spaces
    Apps, many built in Gradio, including Leaderboards
        - It's like an "App Store" for AI demos.
        - Each Space is a public web app built with tools like Gradio, Streamlit, or Static HTML.
        - You can interact with models immediately, without needing to write code or set up environments.

#### HuggingFace Libraries
And the astonishing leg up we get from them

- hub
    ✅ peft - Parameter-Efficient Fine-Tuning
- datasets
    ✅ trl - Reinforcement Learning from Human Feedback
    ✅ sfl - Supervised Fine-Tuning
- transformers
    ✅ accelerate - for distributed training and inference

#### HuggingFace 
https://huggingface.co/
https://huggingface.co/models
https://huggingface.co/datasets
https://huggingface.co/spaces

https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory
https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On
https://huggingface.co/spaces/ed-donner/outsmart

##### HuggingFace Token
https://huggingface.co/settings/tokens

#### Code on a powerful GPU box 
Google Colab
- Run a Jupyter notebook in the cloud with a powerful runtime
- Collaborate with others
- Integrate with other Google services

Runtimes
1. CPU based
2. Lower spec GPU runtimes for lower cost
3. Higher spec GPU runtimes for resource intensive runs

https://colab.research.google.com/

https://colab.research.google.com/drive/18S8gmcFJZatl9BIIzWZaFYb3tgubUz2M#scrollTo=_oS1KxV8GKfl

```code
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)
prompt = "A futuristic class full of students learning AI coding in the surreal style of Salvador Dali"

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator
).images[0]

image.save("surreal.png")
```

```note
# The above code is a simple example of how to use the HuggingFace library to generate an image using a pre-trained model.

使用 Hugging Face 上的 FluxPipeline 文字轉圖像模型，透過 PyTorch 及 CUDA GPU 執行，產生一張「超現實風格 AI 教室」的圖像，並儲存為 surreal.png。
```

#### What you can now do
- Confidently code with Frontier Models
- Build a multi-modal AI Assistant with Tools
- Navigate the HuggingPlace platform; run code on Colab ***

### Week 3 Day 2
#### Learning Objectives
- Understand the 2 different levels of HuggingFace API
- Use pipelines for a wide variety of AI tasks
- Use pipelines to generate text, images and audio

#### The Two API Levels of Hugging Face
- Pipelines
    Higher level APIs to carry out standard tasks incredibly quickly (PoC)
- Tokenizers and Models
    Lower level APIs to provide the most power and control (Production)

#### Pipelines are incredibly versatile and simple
Unleash the power of open-source models in your solutions in 2 lines of code, 常見用途（Icons + 功能）：
- 🙂😠 Sentiment analysis
- 🗂️ Classifier
- 🧾🧑‍⚖️ Named Entity Recognition
- ❓🅰️ Question Answering
- 📄📄📄 Summarizing
- 🔁🌐 Translation
- 🗣️ Text to Speech
- 🎤 Speech to Text

Use pipelines to generate content
- 📝 Text
- 🖼️ Image
- 🎙️ Audio

Google Drive - Notebook (Week 3 day 2 - pipelines.ipynb)
https://colab.research.google.com/drive/1I3K2EzWC5MGyJlGEcmOMrTmHE4tVewbU#scrollTo=vgG4kcT_4lO_

#### What you can now do
- Confidently code with Frontier Models
- Build a multi-modal AI Assistant with Tools
- Use HuggingFace pipelines for a wide variety of inference tasks

### Week 3 Day 3
#### Learning Objectives
- Create tokenizers for models
- Translate between text and tokens
- Understand special tokens and chat templates

#### Introducing the Tokenizer
Maps between Text and Tokens for a particular model
- Translates between Text and Tokens with encode() and decode() methods '<|begin_of_text|>'
- Contains a Vocab that can include special tokens to signal information to the LLM, like start of prompt
- Can include a Chat Template that knows how to format a chat message for this model

| 項目    | Tokenizer        | Embedding                     |
| ----- | ---------------- | ----------------------------- |
| 功能    | 將文字轉換為 token IDs | 將 token IDs 轉換為語意向量           |
| 處理對象  | 文字（text）         | token ID（整數）                  |
| 是否可訓練 | 否（通常是預設詞彙表）      | 是（模型訓練時更新）                    |
| 範例工具  | `AutoTokenizer`  | Transformer 模型內部的 embedding 層 |
| 模型中位置 | 模型外部（預處理）        | 模型內部（第一層）                     |
| 輸入格式  | 文字（string）         | token ID（整數）                  |
| 輸出格式  | token ID（整數）         | 語意向量（向量）                    |
| 轉換方式  | 將文字轉換為 token ID      | 將 token ID 轉換為語意向量           |

#### The Tokenizers for key models
We will experiment with tokenizers for a variety of open-source models

- Llama 3.1 
    Meta led the way
- Phi 3
    Microsoft's entrant
- Qwen2
    Leader from Alibaba Cloud
- Starcoder2
    Coding model

```Note
1. Hugging Face Token with Write access
2. Hugging Face Models need to be authorized in advance via Web Site.
```

#### Pipeline Code
http://localhost:8888/lab/tree/week3/day2.pipelines.ipynb
https://colab.research.google.com/drive/1I3K2EzWC5MGyJlGEcmOMrTmHE4tVewbU  (T4 GPU)

#### Tokenizer Code
http://localhost:8888/lab/tree/week3/day3.tokenizers.ipynb
https://colab.research.google.com/drive/1Lvmx-_XVQ1ntldGWBY2NIfWlV-3pyvV0#scrollTo=y7LTUIlD9Gdm

### Week 3 Day 4
#### Learning Objectives (Models)
- Work with HuggingFace lower level APIs
- Use HuggingFace models to generate text
- Compare the results across 5 open source models

#### We will use these models
We will try Llama 3.1, Phi and Gemma, and you should try Mixtral and Qwen2

- Llama 3.1 from Meta
- Phi 3 from Microsoft
- Gemma from Google
- Mixtral from Mistral (try)
- Qwen 2 from Alibaba Cloud (try)

We will also cover…

- Quantization
| 精度   | 類型說明               | 特點                          |
|--------|------------------------|-------------------------------|
| FP32   | 32-bit float           | 高精度、較耗資源              |
| INT8   | 8-bit integer          | 最常見量化精度，速度快        |
| INT4   | 4-bit integer          | 精度犧牲更多，節省更多空間    |
| BF16   | bfloat16（Google）     | 保留範圍，但減少精度          |

```code
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", quantization_config=quant_config)
```
- Model Internals
    - 有助於**微調（Fine-tuning）或壓縮模型（Quantization）**時做出最佳決策。
    - 可讓工程師針對特定任務或瓶頸進行最佳化。
    - 是打造更高效 Agent 或 AI Assistant 的基礎。
- Streaming

#### Model Code
http://localhost:8888/lab/tree/week3/day4.models.ipynb
https://colab.research.google.com/drive/1l9sEZswOwqOdgzrin0s9hm4zmFkpOtUQ

### Week 3 Day 5
#### Learning Objectives
- Confidently work with tokenizers and models
- Run inference on open-source models
- Implement an LLM solution combining Frontier and open-source models

#### Create a solution that makes meeting minutes
Take an audio recording of a meeting, and generate minutes and actions

- Use a Frontier model to convert the audio to text
- Use the open source model to generate minutes
- Stream back results and show in Markdown

Dataset => https://huggingface.co/datasets/huuuyeah/meetingbank
Download => https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main

MyDrive/
└── llms/
    └── denver_extract.mp3


http://localhost:8888/lab/tree/week3/day5.meeting_minutes.ipynb
https://colab.research.google.com/drive/1klKAzdsW0sa7NUMvX9UHSyDuJwgCsha3

#### Generating Synthetic Data

- Write models that can generate datasets
- Use a variety of models and prompts for diverse outputs
- Create a Gradio UI for your product

#### Week 3 Day 5 - Exercise
day5.meeting_minutes.ipynb => With Gradio UI

### Week 4 Day 1
#### Learning Objectives
- Discuss how to select the right LLM for the task
- Compare LLMs based on their basic attributes and benchmarks
- Use the Open LLM Leaderboard to evaluate LLMs

#### How to compare LLMs
Importantly, LLMs need to be evaluated for suitability for a given task

- Start with the basics
    Parameters
    Context length
    Pricing

- Then look at the results
    Benchmarks
    Leaderboards
    Arenas

#### COMPARING LLMs
The Basics (1) - Compare the following features of an LLM:

- Open-source or closed
- Release date and knowledge cut-off
- Parameters
- Training tokens
- Context length

The Basics (2) - Compare the following features of an LLM:

- Inference cost
    API charge, Subscription or Runtime compute
- Training cost
- Build cost
- Time to Market
- Rate limits
- Speed
- Latency
- License

#### The Chinchilla Scaling Law
Number of parameters ~ proportional to the number of training tokens

- If you're getting diminishing returns from training with more training data, then this law gives you a rule of thumb for scaling your model.
- And vice versa: if you upgrade to a model with double the number of weights, this law indicates your training data requirement.
- The Chinchilla Scaling Law is a guideline for scaling LLMs, suggesting that the number of parameters should be proportional to the number of training tokens.

| 模型              | 參數數量 | 訓練 token 數量     | 是否符合 Chinchilla 法則？ |
| --------------- | ---- | --------------- | ------------------- |
| **GPT-3**       | 175B | 300B tokens     | ❌ 訓練資料略偏少           |
| **Chinchilla**  | 70B  | **1.4T tokens** | ✅ 完全符合              |
| **LLaMA 2 13B** | 13B  | 1.4T tokens     | ✅ 適中偏多              |
| **GPT-4 (推估)**  | \~1T | \~10T+ tokens?  | ✅ 微調訓練資料比例          |

🧪 結論說明：
- GPT-3 使用了大量參數但相對較少的訓練資料，造成推理效果不如預期。
- Chinchilla（由 DeepMind 提出）是以 降低參數量但增加資料量 為目標，結果在多個基準測試中 超越 GPT-3。
- 後來的 LLaMA 與 GPT-4 等模型皆 朝向此法則設計，避免模型「過大卻學不多」的低效現象。

🎯 類比說明：
你可以把 LLM 想像成一個腦袋：
- 參數越多 → 腦容量越大 → 可學習越多樣的語言知識與推理技巧
- 但如果沒有足夠的「訓練資料」去教它，這個大腦也會「笨笨的」（這就是 Chinchilla Scaling Law 要解決的問題）

#### 7 common benchmarks that you will often encounter 你經常會遇到的 7 個常見基準測試）
| Benchmark      | What’s being evaluated | Description                                                                |
| -------------- | ---------------------- | -------------------------------------------------------------------------- |
| **ARC**        | Reasoning**              | A benchmark for evaluating scientific reasoning; multiple-choice questions |
| **DROP**       | Language Comp          | Distill details from text then add, count or sort                          |
| **HellaSwag**  | Common Sense           | "Harder Endings, Long Contexts and Low Shot Activities"                    |
| **MMLU**       | Understanding          | Factual recall, reasoning and problem solving across 57 subjects           |
| **TruthfulQA** | Accuracy               | Robustness in providing truthful replies in adversarial conditions         |
| **Winogrande** | Context                | Test the LLM understands context and resolves ambiguity                    |
| **GSM8K**      | Math                   | Math and word problems taught in elementary and middle schools             |

**Reasoning（推理）是指模型運用邏輯、規則、或知識來進行推斷、分析、判斷的能力。在大型語言模型（LLM）中，這項能力決定了它是否能正確回答需要多步邏輯推理的問題。

#### 3 specific benchmarks

| **Benchmark** | **What's being evaluated** | **Description**                                                           |
| ------------- | -------------------------- | ------------------------------------------------------------------------- |
| **ELO**       | Chat                       | Results from head-to-head face-offs with other LLMs, as with ELO in Chess |
| **HumanEval** | Python Coding              | 164 problems writing code based on docstrings                             |
| **MultiPL-E** | Broader Coding             | Translation of HumanEval to 18 programming languages                      |

#### Limitations of Benchmarks
- Not consistently applied
- Too narrow in scope
- Hard to measure nuanced reasoning
- Training data leakage
- Over-fitting

And a new concern, not yet proven
- Frontier (先進的) LLMs may be aware that they are being evaluated (考前偷看考題)

避免「Frontier LLMs 在被評估時早已見過測試資料」的問題（即 評估失真），可以考慮以下幾種方法：
✅ 1. 使用隱藏測試集（Private Test Sets）
- 建立一組從未公開過的測試資料，確保模型在訓練期間無法接觸到。
- 這類資料不能出現在網路上，也不能來自常見的 benchmark 數據集。

✅ 2. 建立合成測驗（Synthetic Benchmarks）
- 用其他模型或人類設計全新問題，模仿 benchmark 題型但內容不同。
- 確保這些新資料不可能是訓練語料的一部分。

✅ 3. 加入防洩漏檢查（Data Leakage Detection）
- 檢查模型訓練語料是否包含你要用來測試的資料。
- 有些工具（如 DLC）可以協助檢查語料重複或重疊。

✅ 4. 分析模型反應以識別記憶痕跡
- 如果模型回答得過於流暢、準確、快速，可能就是「背過」了答案。
- 可藉由讓模型多次回覆同一問題並觀察變異性，來判斷是否為背誦。

✅ 5. 結合不同評估方式
- 不要只用選擇題（multiple choice），也用開放型問題（open-ended）、推論題（reasoning）、代碼生成等方式混合測試。
- 多樣的題型能更好地辨別「真正理解」與「死記答案」。

✅ 6. 避免過度微調 Benchmark 題目
- 若模型經過針對 ARC、MMLU、HellaSwag 等 benchmark 題目的微調，就會變得「知道」怎麼應考。
- 評估時應使用基礎版本（未特別針對 benchmark 調整過的模型）。

✅ 7. 使用 Blind 評分
- 在人類評分時隱藏是哪個模型產生的回答，防止偏見。
- 這方法常用於 Arena 模式評比，例如 LMSYS Chatbot Arena。

#### 6 Hard, Next-Level Benchmarks

| **Benchmark** | **What's being evaluated** | **Description**                                                                                                        |
| ------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **GPQA**      | Graduate Tests             | 448 expert questions; non-PhD humans score 34% even with web access (Claude 3.5)                                                   |
| **BBHard**    | Future Capabilities        | 204 tasks believed beyond capabilities of LLMs (no longer valid, due to improved LLMs!)                                                            |
| **Math Lv 5** | Math                       | High-school level math 'competition' problems                                                                           |
| **IFEval**    | Difficult instructions     | Like, "write more than 400 words" and "mention AI at least 3 times"                                                    |
| **MuSR**      | Multistep Soft Reasoning   | Logical deduction, such as analyzing 1,000 word murder mystery and answering: "Who has means, motive and opportunity?" |
| **MMLU-PRO**  | Harder MMLU                | A more advanced and cleaned up version of MMLU including choice of '10' answers instead of 4                             |

#### 6 Hard, Next-Level Benchmarks 舉例說明：

✅ 1. GPQA（Graduate-Level Physics QA）
- 測試內容：研究所級別的物理與科學問答。
- 例子：
問：「量子力學中的不確定性原理是什麼？請給出數學式與物理意涵。」
- 應用情境：用於測試 LLM 是否具備高階學術理解與推理能力。

✅ 2. BBHard（Beyond the Benchmarks Hard）
- 測試內容：原本被認為超出 LLM 能力範圍的推理與理解任務。
- 例子：
給一段抽象科幻小說，要求分析其社會隱喻與邏輯前提。
- 應用情境：檢驗 Frontier LLM 在創造力與開放式問題處理上的極限。

✅ 3. Math Lv 5
- 測試內容：高中數學競賽級問題，需高階邏輯與公式運用。
- 例子：
「若一個函數滿足 f(x+y)=f(x)f(y)，且 f(0)=1，求 f(2) 的可能值。
- 應用情境：可應用於 AI 教學輔助或工程問題推演。

✅ 4. IFEval（Instruction Following Evaluation）
- 測試內容：測試模型是否能精準依照複雜指令生成結果。
- 例子：
指令：「寫一篇不少於 400 字的短文，且至少 3 次提到 AI，語氣需為懷舊風格。」
- 應用情境：對話代理或任務型 AI 的效能評估。

✅ 5. MuSR（Multistep Soft Reasoning）
- 測試內容：多步驟的柔性邏輯推理。
- 例子：
「閱讀一篇 1000 字的推理小說片段，推斷誰是兇手並解釋其動機與手段。」
- 應用情境：法律助理 AI、記者助手、偵查分析工具等。

✅ 6. MMLU-PRO（Massive Multitask Language Understanding - PRO版）
- 測試內容：更高階、更乾淨的多任務理解測試，答案選項從 4 個擴增到 10 個。
- 例子：
「請在 10 個選項中選出描述細胞分裂週期順序正確的一項。」
- 應用情境：醫學、工程、法學、教育等專業領域的模型能力驗證。

#### Hugging Face Open LLM Leaderboard
https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?official=true

#### What you can now do
- Code with Frontier Models including AI Assistants with Tools
- Build solutions with open-source LLMs with HuggingFace transformers
- Compare LLMs to identify the right one for the task at hand

### Week 4 Day 2
#### Learning Objectives
- Navigate the most useful leaderboards and Arenas to evaluate LLMs
- Give real-world use cases of LLMs solving commercial problems
- Confidently choose LLMs for your projects

#### Six Leaderboards
A tour of the essential leaderboards for selecting LLMs*
* **HuggingFace Open LLM**
  *New and old version*
  - https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?official=true

* **HuggingFace BigCode**
  *Code generation*
  - https://huggingface.co/bigcode
* **HuggingFace LLM Perf**
* **HuggingFace Others**
  *e.g. Medical, Language-specific*
* **Vellum**
  *includes API cost and context window*
* **SEAL**
  *expert skills*

#### The Arena
https://lmarena.ai/?leaderboard

The LMSYS Chatbot Arena is an amazing resource
- Compare Frontier and Open-source models directly
- Blind human evals based on head-to-head comparison
- LLMs are measured with an 'ELO' rating

Participate in the voting - it's a terrific way to learn about different models while adding to the ecosystem

#### BigCode Leaderboard
https://www.bigcode-project.org/
https://huggingface.co/bigcode
- StarCoder2
- Code Llama2

| 對象類別           | 用途說明                                                                                         |
|:-------------------|:-------------------------------------------------------------------------------------------------|
| AI 研究人員        | 研究程式碼生成、訓練效率、模型解釋性等相關主題，使用 StarCoder 模型與 The Stack 資料集進行實驗。 |
| 程式開發者與工程師 | 使用 StarCoder 等模型進行程式碼補全、自動測試生成、程式重構建議等開發工作。                      |
| 開源貢獻者與社群   | 貢獻開源模型與資料集，協助去敏與資料清理、提升模型品質。                                         |
| 教育與學術機構     | 用於教授生成式 AI 與大型語言模型的程式碼應用，支持學生實作與探索。                               |
| 企業與初創公司     | 整合至內部開發流程，打造客製化的程式開發助理與自動化工具。                                       |

#### Commercial Use Cases
- Law 
    Harvey
- Talent 
    nebula.io
- Porting code
    Bloop (bloop.ai)
- Healthcare 
    Salesforce Health
- Education 
    Khanmigo

| 應用領域                   | 公司 / 產品               | 說明                                           |
| ---------------------- | --------------------- | -------------------------------------------- |
| ⚖️ 法律（Law）             | **Harvey**            | 為律師或法律事務所提供 AI 法律助理，例如自動草擬合約、總結案例、回答法律問題     |
| 🌟 人才招募（Talent）        | **nebula.io**         | 協助 HR 或獵頭快速分析履歷、推薦職缺、撰寫招募信或進行候選人篩選           |
| 💻 程式轉換（Porting Code）  | **Bloop**             | 協助開發者閱讀舊程式碼、轉換語言（從 COBOL 到 Java）、生成文件與單元測試 |
| ❤️‍🩹 醫療保健（Healthcare） | **Salesforce Health** | 提供 AI 醫療助理，例如病歷摘要、自動回答病人查詢、臨床決策支援            |
| 🎓 教育（Education）       | **Khanmigo**          | 由可汗學院開發，提供學生 AI 教學助理，能即時解釋概念、批改作業或模擬老師角色     |

#### BigCode Leaderboard
https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard

#### LLM Perf Leaderboard
https://huggingface.co/spaces/optimum/llm-perf-leaderboard
https://huggingface.co/spaces/optimum/llm-perf-leaderboard#/?official=true

#### HuggingFace Spaces Leaderboard
https://huggingface.co/spaces?q=Leaderboard

#### Open Medical LLMs Leaderboard
https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard

#### Vellum Leaderboard
https://www.vellum.ai/llm-leaderboard#compare
https://www.vellum.ai/llm-leaderboard#/?official=true

#### Chatbot Arena (Human Eval)
https://lmarena.ai/?leaderboard&gad_source=1&gad_campaignid=21946979971&gbraid=0AAAAA-d12XORV-SaxIcoX0ZeRT4lziwkW&gclid=Cj0KCQjwrPHABhCIARIsAFW2XBOMPw3jS-oz9OxTULiQwBdbhMKA-HtA9szFVmRnq7BHns_4A96jO_saAj7oEALw_wcB

- Language Models
    Category
- Overview
- Price Analysis

- Vote
    https://lmarena.ai/

#### Other AIs solutions
| Website              | AI Use Case                                          |
| -------------------- | ---------------------------------------------------- |
| **Harvey AI**        | https://www.harvey.ai Legal assistant for law firms and corporate counsel  |
| **Nebula.io**        | Knowledge AI trained on personal docs                |
| **Bloop.ai**         | AI code search and understanding for devs            |
| **Einstein Copilot** | CRM assistant for business productivity (Salesforce) https://www.salesforce.com/news/press-releases/2024/02/27/einstein-copilot-news/ |
| **Khanmigo.ai**      | AI tutor for students and teachers                   |

#### BUSINESS CHALLENGE
Introducing our commercial challenge this week

- Build a product that converts Python code to C++ for performance
    Solution with a Frontier model
    Solution with an Open-Source model

Let's start by selecting the LLMs most suited for the task

#### What you can now do
- Code with Frontier Models including AI Assistants with Tools
- Build solutions with open-source LLMs with HuggingFace transformers (i.e. from transformers import pipeline, tokenizer, model)
- Confidently choose the right LLM for your project, backed by metrics (Leaderboards, Benchmarks, ELO ratings)
