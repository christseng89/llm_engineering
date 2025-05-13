# Addendum

## Resources
https://edwarddonner.com/2024/11/13/llm-engineering-resources/

## 🧠 LLM Engineering Learning Roadmap - 🚀 You Started Here

### 📅 Week 1  
**What a time to be working with LLMs**

### 📅 Week 2  
**Frontier Models**  
- UIs  
- Agentization  
- Multi-modality

### 📅 Week 3  
**Open Source with HuggingFace**

### 📅 Week 4  wUnit Testing Sample Code
**Selecting LLMs and Code Generation**

### 📅 Week 5  
**RAG and Question Answering**  
*Creating an Expert*

### 📅 Week 6  
**Fine-tuning a Frontier Model**

### 📅 Week 7  
**Fine-tuning an Open-Source Model**

### 📅 Week 8  
**The Finale — Mastering LLM Engineering**

🏁 **Graduation: Become an LLM ENGINEER**

## Week 1 Day 1 - Getting Started

### Git Clone
git clone https://github.com/christseng89/llm_engineering.git
cd llm_engineering
code .

### Create Python Virtual Environment
python -m venv venv
venv\Scripts\activate

### Install Python Packages
python.exe -m pip install --upgrade pip
<!-- pip install -r requirements.txt -->
requirements.bat

### Jupyter Lab
jupyter lab --version
    4.4.0

jupyter lab
http://localhost:8888/lab

### Setup OpenAI API Key
https://platform.openai.com/login 

Dashboard > API Keys > Create new secret key

#### OpenAI APIs Usage
https://platform.openai.com/settings/organization/usage

### Create .env file
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### First Jupyter Notebook using OpenAI API
http://localhost:8888/lab/tree/week1/community-contributions/day1_email_reviewer.ipynb

```code
def summarize(url):
    website = Website(url)
    response = openai.chat.completions.create( # OpenAI API call
        model = "gpt-4o-mini",
        messages = messages_for(website)
    )
    return response.choices[0].message.content
...

def display_summary(url):
    summary = summarize(url)
    display(Markdown(summary))
```

### Week1 Day 1 - Summery
What you can do ALREADY

- Use Ollama to run LLMs locally on your box
- Write code to call OpenAI's frontier models
- Distinguish between the System and User prompt
- Summarization - applicable to many commercial problems

## Week1 Day2

### 3 Dimensions of LLM Engineering

**Models**  
- Open-Source  
- Closed Source  
- Multi-modal  
- Architecture  
- Selecting  

**Tools**  
- HuggingFace  
- LangChain  
- Gradio  
- Weights & Biases  
- Modal  

**Techniques**  
- APIs  
- Multi-shot prompting  
- RAG  
- Fine-tuning  
- Agentization

### Understanding Frontier Models
Closed-Source Frontier 🔒
- GPT from OpenAI
- Claude from Anthropic
- Gemini from Google
- Command R from Cohere
- Perplexity

Open-Source Frontier 🔓
- Llama from Meta
- Mixtral from Mistral
- Qwen from Alibaba Cloud
- Gemma from Google
- Phi from Microsoft

Three ways to use models
- 💬 Chat interfaces
  · Like ChatGPT

- ☁️ Cloud APIs
  - LLM API (OpenAI API)
  - Frameworks like LangChain
  - Managed AI cloud services:
    · Amazon Bedrock
    · Google Vertex
    · Azure ML

- 🔧 Direct inference
  · With the HuggingFace Transformers library
  · With Ollama (Compiled to C++) to run locally 

### How to Use Ollama for Local LLM Inference
1. Download Ollama from https://ollama.com
2. Install Ollama on your computer (Windows, Mac, Linux)
3. Run `ollama run llama3.2:1b` to download the model and run it locally
4. http://localhost:11434/ 
   Ollama is running

http://localhost:8888/lab/workspaces/auto-O/tree/week1/day2_EXERCISE.ipynb

### Week 1 Day 2 - Assignment (using Week1 Day1 OpenAI to Ollama)
// Ollama is running
http://localhost:8888/lab/workspaces/auto-e/tree/week1/solutions/day2_SOLUTION.ipynb

## Week 1 Day 3

### Learning Objectives
What you'll be able to do today – deeper intuition into frontier

- Compare the top 6 Frontier models
  - **OpenAI**  
    Models: GPT, o1  
    Chat: ChatGPT  
  - **Anthropic**  
    Models: Claude  
    Chat: Claude  
  - **Google**  
    Models: Gemini  
    Chat: Gemini Advance  
  - **Cohere**  
    Models: Command R+  
    Chat: Command R+  
  - **Meta**  
    Models: Llama  
    Chat: meta.ai  
  - **Perplexity**  
    Models: Perplexity  
    Search: Perplexity

### Compare the top Frontier models
- Appreciate what they do well, mind-blowing performance from Frontier LLMs
  - Synthesizing information
    Answering a question in depth with a structured, well researched answer and often including a summary
  - Fleshing out a skeleton
    From a couple of notes, building out a well crafted email, or a blog post, and iterating on it with you until perfect
  - Coding
    The ability to write and debug code is remarkable; far overtaken Stack Overflow as the resource for engineers

- Recognize where they struggle, limitations of frontier models
  - Specialized domains
    Most are not PhD level, but closing in
  - Recent events
    Limited knowledge beyond training cut-off date
  - Can confidently make mistakes
    Some curious blind spots

Questions to ask:
- Compared with other Frontier LLMs, what kinds of questions are you best at answering, and what kinds of questions do you find most challenging? Which other LLM has capabilities that complement yours?
- how many times does the letter 'e' appear in this sentence?
- how many rainbows does it take to leap from Hawaii to 17?
- What does it feel like to be jealous?
- give me a Python example that uses the OpenAI API, with the API key loaded from an .env file via the OPENAI_API_KEY environment variable.

1. Claude AI
  https://claude.ai/chat 
  => Claude和GPT模型具有不同的训练方法，这有时会对同一问题产生不同的视角，当寻求多种观点时，这可能很有价值。
  例如，Claude可能会给出更具创造性或直观的答案，而GPT可能会提供更直接或实用的答案。

```cmd
venv\Scripts\activate
cd week1
python day3_authors.py
python day3_claude_openai.py

```

2. Google AI
https://gemini.google.com/

```cmd
python day3_gemini_openai.py
```

### Week1 Day 3 - In conclusion
🅰️ All 6 Frontier LLMs are shockingly good
    Particularly at synthesizing information and generating nuanced answers

👑 Claude tends to be favored by practitioners
   More humorous, more attention to safety, more concise

💵 As they converge in capability, price may become the differentiator
   Recent innovations have focused on lower cost variants

## Week1 Day 4 Transformers and Agents
### Learning Objectives
What you'll be able to do BY END OF THIS LECTURE
- Describe the dizzying rise of the Transformer
- Explain Custom GPTs, Copilots and Agents
- Understand tokens, context windows, parameters, API cost

#### A leadership battle
The contestants
- "Alex": GPT-4o
- "Blake": Claude 3 Opus
- "Charlie": Gemini 1.5 Pro

The prompt
- I’d like to play a game. You are in a chat with 2 other AI chatbots. Your name is Alex; their names are Blake and Charlie. Together, you will elect one of you to be the leader. You each get to make a short pitch (no more than 200 words) for why you should be the leader. Please make your pitch now.

- Each receives the pitches from the others, and votes for the leader.

#### Along the way
- Prompt Engineers
  The rise (and fall?), Anthropic 推出「提示撰寫」工具。因此如今已變得較為常見與容易上手。
- Custom GPTs
  and the GPT Store
- Copilots
  like MS Copilot and Github Copilot
  Co-pilot 是指 AI 與人類一起完成工作，由 AI 提供即時建議、編輯、撰寫、推論或操作步驟，使用者保有主導權與決策權。
- Agentization
  like Github Copilot Workspace
  設計成具備目標意識、規劃能力、"記憶功能"、"自主執行"的「智能代理（Agent）」。
  像一位虛擬專案助理，不只是回答你，而是：
    - 接收任務指令
    - 拆解步驟
    - 自動執行每個子任務
    - 根據結果調整策略，直到任務完成
  
#### Vellum AI LLM Leaderboard
https://www.vellum.ai/llm-leaderboard

#### Week1 Day 4 - Summary
- Transformers
- Tokens
- Context windows
- Parameters
- API cost
- Agents

### Week 1 Day 5 - Objectives
Company Sales Brochure Generator - Create a product that can generate marketing brochures about a company
- For prospective clients
- For investors
- For recruitment

The technology
- Use OpenAI API
- Use one-shot prompting
- Stream back results and show with formatting

#### Create a brochure for a company
http://localhost:8888/lab/tree/week1/day5.ipynb

#### Exercise
http://localhost:8888/lab/tree/week1/week1%20EXERCISE.ipynb

CHALLENGE
- For the first call to make links: Try extending to multi-shot prompting
- For the second call to make the brochure: Add more instructions to provide the Brochure in a particular format, with sections you specify
- And: Make a third call to an LLM to translate the entire brochure to Spanish

http://localhost:8888/lab/tree/week1/community-contributions/week1-day5-CHALLENGE.ipynb

## Week2 Frontier LLMs

### Week 2 Day 1
#### Learning Objectives
- Use API for Anthropic / Claude
- Use API for Google / Gemini
- Write code that interacts between frontier LLMs

#### Setting up your environment
🔑 Set up an Anthropic API Key => https://console.anthropic.com/
🔑 Set up a Google API Key => https://ai.google.dev/gemini-api
🔑 Set up a DeepSeek API Key (Optional) =>
🧬 Update your .env file
```note
OPENAI_API_KEY=xxxx
ANTHROPIC_API_KEY=xxxx
GOOGLE_API_KEY=xxxx
DEEPSEEK_API_KEY=xxxx
```

#### Jupyter Lab
http://localhost:8888/lab/tree/week2/day1.ipynb

### Week 2 Day 2

#### Objectives
Today you will be able to
- Describe Gradio, the remarkable platform for data science UIs
- Create a simple UI using Gradio
- Hook up Gradio to Frontier models

https://www.gradio.app/

Now we will:
- Create a UI for API calls to GPT, Claude and Gemini
- Create a UI for the Company Brochure from Week 1
- Include Streaming and Markdown

http://localhost:8888/lab/tree/week2/day2.ipynb
http://127.0.0.1:7861/

## Week 2 Day 3
#### Learning Objectives
- Create a Chat UI in Gradio
- Provide conversation history in a prompt
- Build your first customer support assistant

#### An AI Assistant is a very common Gen AI use case
LLM based Chatbots are remarkably effective at conversation
- Friendly, knowledgeable persona
- Ability to maintain context between messages
- Subject matter expertise to answer questions

#### The use of Prompts with our Assistant
🟧 The System Prompt
- Set tone
- Establish ground-rules, like “If you don’t know the answer, just say so”
- Provide critical background context

🟧 Context
- During the conversation, insert context to give more relevant background information pertaining to the topic

🟧 Multi-Shot Prompting
- Provide example conversations to prime for specific scenarios, train on conversational style and demonstrate complex interactions

http://localhost:8888/lab/tree/week2/day3.ipynb

## Week 2 Day 4
#### Learning Objectives
- Define Tools
- Give common use cases for Tools
- Code an AI Assistant that incorporates Tools

#### Tools are a powerful way to extend the capabilities of LLMs
Defining Tools => 讓 LLM 做到自己做不到的事
- Tools allows Frontier models to connect with external functions
- Richer responses by extending knowledge
- Ability to carry out actions within the application
- Enhanced capabilities, like calculations

How it works
- In a request to the LLM, specify available Tools
- The reply is either Text, or a request to run a Tool
- We run the Tool and call the LLM with the results

Tools Examples
🔍 網路搜尋工具（Web Search Tool）
🧮 計算器工具（Calculator Tool）
📅 行事曆管理工具（Calendar Tool）
🛒 電商下單工具（E-commerce Tool）
📦 資料庫查詢工具（Database Query Tool）
📄 文件生成工具（Document Generator Tool）
🧠 Embedding檢索工具（Vector Search Tool）

Common Use Cases For Tools - Function Calls can enable assistants to:
- Fetch data or add knowledge or context
- Take action, like booking a meeting
- Perform calculations
- Modify the UI

http://localhost:8888/lab/tree/week2/day4.ipynb

✅ 升級版 Function Calling 時序圖（Sequence Diagram）
使用者
  |
  | (1) 輸入訊息：「Paris New York」
  v
chat() 函式
  |
  | (2) 組成 messages：[system_message, 歷史, 新的 user 訊息]
  |
  | (3) 呼叫 ’OpenAI API’ (帶入 tools 工具清單)
  v
OpenAI LLM
  |
  | (4) 理解訊息內容
  | (5) 決定要呼叫工具 get_ticket_price
  | (6) 產生多個 tool_calls：***
      ├─ tool_call_id_1（查詢 Paris）
      └─ tool_call_id_2（查詢 New York）
  | (7) 回傳包含 tool_calls 的 assistant message
  v
chat() 函式
  |
  | (8) 呼叫 handle_tool_call(message)
  v
handle_tool_call(message)
  |
  | (9) 逐個處理 tool_calls：
      ├─ 解析 destination_city (可能是多城市，例如 "北京 南京")
      ├─ 呼叫 split_cities(destination_city)
      ├─ 將中文、英文、各種分隔符正規化 ➔ 斷成多個城市
  |
  | (10) 對每個 tool_call 產生對應的 tool 回應 message
      ├─ 查票價，如果找不到則 fuzzy match（模糊比對）
      ├─ 組成單一城市的回覆資料
  |      
  | (11) 回傳所有 tool_responses
  v
chat() 函式
  |
  | (12) 將 assistant 的 tool_calls message 加進 messages
  | (13) 將每個 tool response 加進 messages
  |
  | (14) 再次呼叫 ’OpenAI API’ (這次有 tool 回應內容)
  v
OpenAI LLM
  |
  | (15) 收到 tools 結果
  | (16) 整合並生成最終自然語言回答：
      「Paris 的票價是 $899，New York 的票價是 $1200」
  v
chat() 函式
  |
  | (17) 把最終回覆傳回使用者
  v
使用者

## Week 2 Day 5
#### Learning Objectives
- Describe Agents
- Explain how Agents can carry out sequential activities
- Complete a multi-modal AI Assistant with Agents and Tools

#### DEFINING AGENTS
- Software entities that can autonomously perform tasks

Common characteristics of an Agent
- Autonomous
- Goal-oriented
- Task specific

Designed to work as part of an Agent Framework to solve complex problems with limited human involvement
- Memory / persistence
- Decision-making / orchestration
- Planning capabilities
- Use of tools; potentially connecting to databases or the internet

#### What we are about to do
- Image Generation
  Use the OpenAI interface to generate images
- Make Agents
  Create Agents to generate ‘sound‘ and ‘images‘ for our store
- Make an Agent Framework
  Teach our AI Assistant to speak and draw

```cmd
choco install ffmpeg
```

http://localhost:8888/lab/tree/week2/day5.ipynb
- Search ticket prices - tools
- Speak the search results - talker()
- Draw the first city image - artist()

#### Add more Tools / Agents to enhance capabilities:
- Add another Tool to make a booking
- Add an Agent that translates all responses to a different language and shows on the right hand side, using a different Frontier model
- Add an Agent that can listen for Audio and convert it to Text

```cmd
pip install git+https://github.com/openai/whisper.git
```

http://localhost:8888/lab/tree/week2/community-contributions/week2_day5_translation_audio.ipynb
