# Addendum

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
