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

http://localhost:8888/lab/tree/week5/community-contributions/day5.markdown_llm_knowledge.ipynb
