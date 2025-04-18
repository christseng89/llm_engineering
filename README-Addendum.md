# Addendum

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

Claude AI
https://claude.ai/chat 
  => Claude和GPT模型具有不同的训练方法，这有时会对同一问题产生不同的视角，当寻求多种观点时，这可能很有价值。
  例如，Claude可能会给出更具创造性或直观的答案，而GPT可能会提供更直接或实用的答案。

```cmd
venv\Scripts\activate
cd week1
python day3_authors.py
python day3_claude_openai.py

```

Google AI
https://gemini.google.com/
