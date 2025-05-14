# LLM Learning NOTES part 3
```cmd
venv\Scripts\activate
jupyter lab
```
### Week 5 Day 1
#### What is RAG
**RAG** stands for **Retrieval-Augmented Generation**, a technique in natural language processing (NLP) that combines two components:

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
