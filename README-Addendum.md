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
