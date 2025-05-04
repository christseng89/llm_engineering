# LLM Learning NOTES part 2
```cmd
venv\Scripts\activate
jupyter lab
```

### Week3 Day1
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

### Week3 Day2
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

### Week3 Day3
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
