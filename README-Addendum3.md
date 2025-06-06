# LLM Learning NOTES part 3
```cmd
venv\Scripts\activate
jupyter lab
```
### Week 5 Day 1
#### What is RAG
**RAG** stands for **Retrieval-Augmented Generation (檢索增強生成)**. It is an advanced technique in natural language processing (NLP) that combines a language model (like GPT) with an external knowledge retrieval system, typically a vector database.
---

##### 🔍 **1. Retrieval:**

* Pulls relevant information from an external **knowledge base** (e.g., documents, databases, or vector stores like FAISS, Pinecone, Weaviate etc.).
* Ensures the language model has access to **up-to-date or domain-specific knowledge** it may not have seen during training.

---

##### ✍️ **2. Generation:**

* Uses a large language model (LLM) like GPT to **generate responses** based on both the user query and the retrieved documents.
* The retrieved content is fed into the prompt (often as context) to **ground the output** in real information.

---

##### ✅ **Why Use RAG?**

* Helps overcome **knowledge cutoff** limitations of static models.
* Improves **accuracy**, especially in specialized domains (e.g., legal, medical, finance).
* Enhances **explainability** by pointing to retrieved sources.

---

##### 🧠 Real-World Example:

If you ask:

> “What’s the latest update in the EU AI Act?”

A standard LLM might guess (limited by its training data), but a RAG system will:

1. **Retrieve** news articles or official updates from a knowledge source.
2. **Generate** a summarized answer grounded in those retrieved documents.

#### ⚙️ **RAG Can Be Built With Tools Like:**

* [LangChain](https://www.langchain.com/)
* [LlamaIndex (formerly GPT Index)](https://www.llamaindex.ai/)
* [Haystack](https://haystack.deepset.ai/)
* Vector databases: FAISS, Pinecone, Weaviate

### Learning Objectives
- Explain the big idea behind Retrieval Augmented Generation (RAG)
- Walk through the high level flow for adding expertise to queries
- Implement a 'toy' version of RAG without vector databases.. yet

#### INTRODUCING RAG - Motivating RAG

We've already used techniques to improve prompts
- Multi-shot prompting
- Use of tools
- Additional context

We can take this to the next level
- Build a database of expert information, called a Knowledge Base
- Every time the user asks a question, search for anything relevant in the Knowledge Base
- Add relevant details to the prompt

#### A small example of the small idea
Business setup
- We work for an Insurance Tech startup
- We have a Knowledge Base of the company shared drive
- Task is to build an AI Knowledge Worker

The Blunt Instrument implementation
- Read names of products and employees
- See if questions refer to employee or products by name
- Add relevant details to the prompt

http://localhost:8888/lab/tree/week5/day1.ipynb

#### Encoding LLMs and Vector Embeddings
##### Auto-Encoding vs Auto-Regressive LLMs
🔁 Auto-Regressive LLMs predict a future token from the past
    - 根據前面的 token 來預測下一個 token
    - 通常用於自然語言生成任務
##### Examples:
- GPT-3GPT（Generative Pre-trained Transformer）系列，如 GPT-3、GPT-4
    👉 例子：輸入 "The movie was surprisingly..." → GPT 會預測下一個詞，如 "good"、"bad"、"funny" 等。
- ChatGPT
    👉 根據你的問題一步步預測回覆語句。

🔁 Auto-Encoding LLMs produce output based on the full input
    - 利用整句輸入的上下文來進行編碼與理解
    - 常用於分類、語義搜尋、特徵提取
    - Applications include Sentiment Analysis and classification
    - Also used to calculate "Vector Embeddings", representing an input as a list of numbers – i.e. a vector
    - Examples include BERT from Google and OpenAIEmbeddings from OpenAI
##### Examples:
- BERT (Bidirectional Encoder Representations from Transformers)
    👉 例子：輸入一句話 "The movie was surprisingly good." → BERT 會整句理解，將其轉換為向量用於情感分類（Positive/Negative）
- OpenAI Embeddings
    👉 將輸入句子轉成向量，再與其他句子的向量比對相似度，用於搜尋或推薦系統。

#### These vectors mathematically represent the 'meaning' of an input
- Can represent a character, a token, a word, an entire document, or something abstract
- Typically have hundreds, or thousands of dimensions
- Represent an 'understanding' of the inputs; similar inputs are close to each other
- Support 'vector math' like the famous example: "King - Man + Woman = Queen"

### Week 5 Day 2
#### Learning Objectives
- Describe the LangChain framework, with benefits and limitations
- Use LangChain to read in a Knowledge Base of documents
- Use LangChain to divide up documents into overlapping chunks

#### INTRODUCING LangChain
LangChain
- OpenSource framework launched in October 2022
- Provides a common framework for interfacing with many LLMs
- Includes its own declarative language: LangChain Expression Language (LCEL)

Pros & Cons
- Greatly simplifies the creation of applications using LLMs (e.g. AI assistants, RAG, summarization) – fast time to market
- Wrapper code around LLMs makes it easy to swap models
- As APIs for LLMs have matured, converged and simplified, the need for a unifying framework like LangChain has decreased

LLM vs LangChain 🔧 說白話一點：
- LLM 是核心引擎，LangChain 是整台車的架構和自動駕駛系統。
- 你可以只用引擎（呼叫 GPT API），但如果你要處理多文件、工具調用、多步推理、記憶、代理等複雜邏輯，就需要 LangChain 來幫你組織。

#### We will now use LangChain to load our Knowledge Base
- Read in the documents in all folders
- Add meta-data to the documents
- Break down the contents into overlapping chunks

http://localhost:8888/lab/tree/week5/day2.ipynb

### Week 5 Day 3 (Vector DBs)
#### Learning Objectives
- Convert our chunks of text into Vectors using OpenAIEmbeddings
- Store the Vectors in Chroma, one of the most popular open-source Vector datastores
- Visualize and explore Vectors in a Chroma Vector Datastore in 2D and 3D

#### Vector embedding models
- word2vec (2013)
- BERT (2018)
- OpenAI Embeddings (2024 updates)

#### Introducing Chroma 
https://www.trychroma.com/

Chroma is the open-source AI application database. Batteries included.
    Embeddings, vector search, document storage, full-text search, metadata filtering, and multi-modal. All in one place. Retrieval that just works. As it should be.

```cmd
pip install -U langchain_chroma
```

http://localhost:8888/lab/tree/week5/day3.ipynb

#### POPULATING THE VECTOR DATASTORE - This is where LangChain shines

- Create the Chroma datastore and populate with Vector Embeddings of our Knowledge Base, in 2 lines of code.
```code
# 1 Use OpenAI Embeddings model
embeddings = OpenAIEmbeddings()

# 2 Create vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=db_name
)

```

### Week 5 Day 4
#### Learning Objectives
- Create a Conversation Chain in LangChain for a chat conversation with retrieval
- Ask questions and receive answers demonstrating expert knowledge
- Build a Knowledge Worker assistant with chat UI

#### Key Abstractions in LangChain
- LLM
- Retriever
- Memory

#### A Conversation Chain with RAG and Memory
```code
# 1 create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# 2 set up the conversation memory for the chat
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True, 
    input_key="question",      # ✅ user query
    output_key="answer"        # ✅ model response
)

# 3 create a retriever from the Chroma datastore
retriever = vectorstore.as_retriever()

# 4 putting it together (LLM, Retriever, Memory)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
)

```

http://localhost:8888/lab/tree/week5/day4.ipynb # CHROMA
http://localhost:8888/lab/tree/week5/day4.5.ipynb # FAISS

### Week 5 Day 5
#### LangChain Expression Language (LCEL)
LCEL is a declarative language that can be used as an alternative to the code approach
- Describe what you want to achieve in a YAML file
- Arguably not much easier than coding directly

```code
variables:
  - name: MODEL
  - name: TEMPERATURE
    default: 0.7
  - name: PERSIST_DIRECTORY
    default: 'vector_db'

components:
  - name: OpenAI_LLM
    type: ChatOpenAI
    parameters:
      temperature: ${TEMPERATURE}
      model_name: ${MODEL}

  - name: ConversationMemory
    type: ConversationBufferMemory
    parameters:
      memory_key: chat_history
      return_messages: true

  - name: OpenAIEmbeddings
    type: OpenAIEmbeddings

  - name: ChromaVectorStore
    type: Chroma
    parameters:
      documents: ${chunks}
      embedding: ${OpenAIEmbeddings}
      persist_directory: ${PERSIST_DIRECTORY}

  - name: VectorStoreRetriever
    type: VectorStoreRetriever
    parameters:
      vectorstore: ${ChromaVectorStore}
      search_kwargs:
        k: 20

  - name: ConversationalChain
    type: ConversationalRetrievalChain
    parameters:
      llm: ${OpenAI_LLM}
      retriever: ${VectorStoreRetriever}
      memory: ${ConversationMemory}

output:
  - name: conversation_chain
    from: ${ConversationalChain}

```

#### Behind the curtain
Understanding how LangChain works, and identifying & fixing common problems

🛠️ Topics:
- Using Callbacks
    To output prompt details
    LangChain 允許你使用 callback（回呼）來觀察每一步在幹什麼，例如：模型收到什麼 prompt、產生什麼 output。
    ```code
    from langchain.callbacks import StdOutCallbackHandler

    handler = StdOutCallbackHandler()
    chain.invoke("你能解釋什麼是LLM嗎？", callbacks=[handler])

    ```
- Diagnosing a common problem
    The right chunks are not being provided
    大多數錯誤不是來自 LLM，而是來自「提供給模型的 chunk（資料片段）不對」。
- Fixing the problem
    Chunking differently; providing more chunks
    使用更合適的 chunking 方法，例如：每段 500 字、或根據語意斷句（而不是固定字數）。
    ```code
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    ```
- Demystifying LangChain
    It's actually not hard to build RAG directly

✅ 小結：
    LangChain 很強大，但最常見的問題其實來自資料切片不對，或不理解內部是怎麼組合的。掌握 callback、chunk 設定、RAG 架構，就能有效除錯和優化。

#### Advanced ideas to take it to the next level
- If you use Google Workspace, use Google's API to read your own docs
- If you use MS Office, use libraries to read Office docs
- Harder - use libraries to connect to your email inbox, and Slack, and more!

## Week 6 LoRA and QLoRA
### Week 6 Day 1
- Prerequisite
  http://localhost:8888/lab/tree/week5/community-contributions/day5.markdown_llm_knowledge.ipynb

#### Learning Objectives
- Download a Dataset from the HuggingFace hub
- Examine a dataset
- Identify evaluation criteria for judging success

#### So far we have focused exclusively on INFERENCE
Techniques to improve results at run-time with Closed and Open-Source models

- Multi-shot prompting
- Prompt Chaining
- Tools / Function calling
- RAG / Knowledge Base

#### This week we turn to TRAINING
- Training a multi-billion parameter model from scratch would cost tens to hundreds of million $
- Instead, we take advantage of 'Transfer Learning', 
- We take a pretrained model as base, and use additional training data to fine-tune it for our task

Transfer Learning
  Take a pretrained model — one that has already learned useful representations from a large dataset — and 'fine-tune' it with your own, often smaller, dataset.

#### A Juicy Commercial Problem
Given a description of a product, predict its price
- For a marketplace to estimate prices of goods
- Future versions should be able to write and improve descriptions too
- We'd typically use a Regression model to predict prices, but there are good reasons to try 'Gen AI'
  - We can train an LLM and evaluate it very clearly
  - It means we can battle with GPT-40
  - Spoiler alert: the frontier models are already great at this!

✅ 什麼是 Gen AI？
Generative AI 是指能夠「生成內容」的 AI 模型，常見的例子包括：

- 生成文字 → ChatGPT、Claude、Gemini
- 生成圖片 → Stable Diffusion、DALL·E
- 生成音訊 → TTS（text-to-speech）
- 生成程式碼 → Code LLMs（如 CodeLlama、CodeGemma）

這類模型是基於深度學習，特別是 Transformer 架構訓練出來的，可以 從 prompt 生成內容。例如：
  - Prompt: "請幫我寫一篇商品描述，關於一款智能咖啡機。"
  - Gen AI 就能自動生成描述內容。

#### Finding datasets
- Your own proprietary data
- Kaggle
- HuggingFace datasets
- Synthetic data
- Specialist companies like Scale.com

#### HuggingFace datasets
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories
- Appliances

#### Digging into the data
We'll do some work

- Investigate
- Parse
- Visualize
- Assess Data Quality
- Curate ***
- Save

http://localhost:8888/lab/tree/week6/day1.ipynb

- MatPlotLib supported colors
  https://matplotlib.org/stable/gallery/color/named_colors.html#base-colors

#### How will we evaluate performance?
From our Predicted Prices versus Actual Prices

- Model-centric or Technical Metrics
  - Training loss
  - Validation loss
  - Root Mean Squared Log Error (RMSLE)

- Business-centric or Outcome Metrics
  - Average price difference
  - % price difference
  - % estimates that are "good"/"bad"

#### What you can now do
- Generate text and code with Frontier Models including AI Assistants with Tools, and with open-source models with HuggingFace transformers
- Confidently choose the right LLM for your project, backed by metrics
- Create advanced RAG solutions with LangChain
- Select, investigate and curate a Dataset

### Week 6 Day 2
#### Learning Objectives
- Lay out a 5 step strategy for selecting, training and applying an LLM
- Contrast the 3 techniques for improving performance
- Give common use cases for each of the techniques

#### 5 Step Strategy
To selecting, training and applying an LLM to a commercial problem

- Understand
- Prepare
- Select (Model)
- Customize (Training)
- Production (Deployment) and Evaluate

#### Customization Options:
🧠 **Prompt** Engineering: Write a system prompt like:
  “You are a helpful HR assistant for ACME Corp. Answer questions using ACME’s HR policy.”

📚 **RAG**:
  Load all ACME’s HR docs into a vector store (e.g. FAISS) and let the model retrieve context.

🧬 **Fine-tune**:
  Train on past HR inquiries + answers to improve accuracy in your tone & structure.

##### First step Understand
Activities:

- Gather business requirements for the task
- Identify performance criteria
  - Particularly the Business Centric metrics
- Understand the data: quantity, quality, format
- Determine non-functional requirements
  - Cost constraints, scalability, latency
  - R&D / build budget and implementation timeline

##### Second step Prepare
Activities:

- Research existing / **non-LLM** solutions
  - Potential baseline model
- Compare relevant LLMs
  - The basics, including context length, price and license
  - Benchmarks, **'Leaderboards'** and **'Arenas'**
  - Specialist scores for the task at hand
- Curate data: clean, preprocess and split
  - **Train**, **Validation** and **Test** sets

📘 Example Scenario:
- Goal: Classify customer complaints by department (e.g. Billing, Support, Technical)

  🔹 Non-LLM: Use a TF-IDF + SVM classifier (fast, interpretable)
  🔹 LLM: Use GPT-4 with prompt tuning or fine-tuning

💡 Why consider non-LLM first?
- Faster to deploy – Often easier to build and test
- Cheaper – No GPU, no token billing
- Easier to explain – Rule-based or classical models can be more interpretable
- Good enough – In many business cases, simple solutions already meet requirements

##### Third step Select
Activities:

- Choose LLM(s)
- Experiment
- Train and validate with curated data

##### Fourth step Customize
3 techniques to optimize the performance of the model

- Prompting
  - multi-shot, chaining (Chain-of-Thought) and tools
- RAG
- Fine-tuning (Training)
  - LoRA and QLoRA

✅ Use Cases of Chaining
| Task Type               | How Chaining Helps                       |
| ----------------------- | ---------------------------------------- |
| Math / logic puzzles    | Breaks into small steps                  |
| Legal or policy QA      | Shows reasoning from rules to conclusion |
| Programming / debugging | Walks through logic to suggest fixes     |
| Business decisions      | Compares options with pros & cons        |

###### 3 Techniques: Pros
- Prompting
  - Fast to implement
  - Low cost
  - Often immediate improvement
- RAG
  - Accuracy improvement with low data needs
  - Scalable
  - Efficient
- Fine-tuning
  - Deep expertise & specialist knowledge
  - Nuance (更有「人味」、「情境感」、「禮貌性」，這就是 Nuance)
  - Learn a different tone / style
  - Faster and cheaper inference

###### 3 Techniques: Cons
- Prompting
  - Limited by context length
  - Diminishing returns
  - Slower, more expensive inference
- RAG
  - Harder to implement
  - Requires up-to-date, accurate data
  - Lacks nuance
- Fine-tuning
  - Significant effort to implement
  - High data needs
  - Training cost
  - Risk of "catastrophic forgetting"
    - 當模型在學習新任務或新資料時，會突然「忘記」之前學過的重要知識。

🛠️ 如何避免 Catastrophic Forgetting？
| 方法                      | 說明                       |
| ----------------------- | ------------------------ |
| ✅ 使用 **LoRA / PEFT** 微調 | 保留原模型核心，覆蓋特定層，減少遺忘風險     |
| ✅ **多任務**混合訓練           | 將原知識和新任務一起訓練             |
| ✅ **Replay** 原始資料       | 在訓練時加入部分原始樣本防止模型「全然改變記憶」 |
| ✅ 使用 **RAG** 替代 Fine-tuning | 用外部知識庫而不是修改模型參數本身        |

###### 3 Techniques: Use Cases
- Prompting
  - Often the starting point for optimizing a project, with a Frontier LLM
- RAG
  - You need high accuracy without the cost of fine-tuning; you have a Knowledge Base
- Fine-tuning
  - You have a specialized task with a high volume of data (datasets), and you need top performance

##### Fifth step Production
Activities:

- Determine API between model and platform(s)
- Identify model hosting and deployment architecture
- Address scaling, monitoring, security and compliance
- Measure the Business-Focused Metrics identified in step 1
- **Continuously** retrain and measure performance

http://localhost:8888/lab/tree/week6/day2.lite.ipynb

- http://localhost:8888/lab/tree/week6/items.py
- http://localhost:8888/lab/tree/week6/loaders.py

📌 Summary of Curation Rules

| Rule          | Condition                                                       |
| ------------- | --------------------------------------------------------------- |
| Price range   | 0.5 ≤ price ≤ 999.49                                            |
| Text length   | ≥ 300 characters                                                |
| Token count   | ≥ 150 tokens after processing                                   |
| Content type  | Must include meaningful `description`, `features`, or `details` |
| Noise removal | Removes known boilerplate and long numeric codes                |
| Final format  | Generates a price-prediction prompt for LLM training            |

### Week 6 Day 3
#### Learning Objectives
- Explain the role of a baseline model
- Create a traditional ML solution with features and linear regression
- Apply more advanced NLP techniques including SVR (Support Vector Regression)

#### The importance of a Baseline
- Start cheap and simple
- Gives a benchmark to improve on
- An LLM might not be the right solution

#### Traditional ML models
We will quickly set up these solutions to give us a starting point

1. Feature engineering & Linear Regression
2. Bag of Words & Linear Regression
3. word2vec & Linear Regression
4. word2vec & Random Forest
5. word2vec & SVR

http://localhost:8888/lab/tree/week6/day3.ipynb

### Week 6 Day 4 - Frontier Models
#### Learning Objectives
- Build a framework to solve a commercial problem using a Frontier model
- Run our test dataset against GPT-4o-mini
- Run our test dataset against Claude-3.5-Sonnet

#### Example of Frontier Models
| Model                | Organization    | Description                                              |
| -------------------- | --------------- | -------------------------------------------------------- |
| **GPT-4 / GPT-4o**   | OpenAI*         | Multimodal, high reasoning and task-solving ability      |
| **Claude 3 Opus**    | Anthropic*      | Strong focus on alignment and reliability                |
| **Gemini 1.5 Ultra** | Google DeepMind | Multimodal and long-context capabilities                 |
| **Command R+**       | Cohere          | Retrieval-augmented generation (RAG) specialized         |
| **LLaMA 3 (70B)**    | Meta            | Open-weight large model, fine-tuned variants widely used |

http://localhost:8888/lab/workspaces/auto-C/tree/week6/day4-results.ipynb

### Week 6 Day 5
#### Learning Objectives
- Understand the process for Fine-Tuning a Frontier model
- Create the fine-tuning dataset and run fine-tuning
- Test a fine-tuned Frontier model
#### Fine-tuning a Frontier model
https://platform.openai.com/finetune/

#### Three Stages to Fine-Tuning with OpenAI
- Create **Training Dataset** in **jsonl** format and upload to OpenAI
  https://platform.openai.com/storage/files
- Run training – training loss and validation loss should decrease
    1. https://wandb.ai => create an API key
    2. https://platform.openai.com/account/organization => add Weights & Biases key from step 1 
- Evaluate results, tweak and repeat
  - https://wandb.ai/samfire5200-china-systems/gpt-pricer?nw=nwusersamfire5200

OpenAI expects data in JSONL format, Rows of JSON each containing messages in the usual prompt format
```code
{"messages": [{"role": "system", "content": "You estimate prices..."}]}
{"messages": [{"role": "system", "content": "You estimate prices..."}]}
{"messages": [{"role": "system", "content": "You estimate prices..."}]}
{"messages": [{"role": "system", "content": "You estimate prices..."}]}
{"messages": [{"role": "system", "content": "You estimate prices..."}]}
{"messages": [{"role": "system", "content": "You estimate prices..."}]}

```

#### Key Objectives of Fine-Tuning for Frontier models (Well Fine-tuned was disappointing!)
- Setting style or tone in a way that can't be achieved with prompting
- Improving the reliability of producing a type of output
- Correcting failures to follow complex prompts
- Handling edge cases
- Performing a new skill or task that's hard to articulate in a prompt

A problem like ours doesn't benefit significantly from Fine Tuning
- The problem and style of output can be clearly specified in a prompt
- The model can take advantage of its enormous world knowledge from its pre-training; providing a few hundred prices doesn't help

✅ 無法透過 Prompt Engineering 解決的資料集或任務，才比較適合用 Fine-Tune 模型來處理。

| 類型         | 適合 Prompt | 適合 Fine-Tune |
| ---------- | --------- | ------------ |
| 翻譯、摘要、文法修正 | ✅ 是       | ❌ 否（已學過）     |
| 一般知識問答     | ✅ 是       | ❌ 無明顯效益      |
| 應用情境變化大    | ❌ 難以穩定學習  | ✅ 視任務而定      |

WEEK 6 **CHALLENGE** FOR YOU: Experiment with larger training sets and more prompt engineering and BEAT THE CURRENT BASELINE

http://localhost:8888/lab/tree/week6/day5-results.ipynb

## Finetune day2 with concurrent model to create datasets
http://localhost:8888/lab/tree/week6/day2.lite1.ipynb
- From 4 hours to 30 minutes
