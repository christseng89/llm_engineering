# LLM Learning NOTES part 4
```cmd
venv\Scripts\activate
jupyter lab
```

## Week 7 - Fine-Tuned Open Source LoRa and QLoRA Models
### Week 7 Day 1
#### Learning Objectives
- Explain LoRA (**Low-Rank Adaptation**) and QLoRA (**Quantization and Low-Rank Adaptation**) for fine-tuning Open Source models
- Describe Quantization and QLoRA
- Explain 3 key hyper-parameters: r, alpha and target modules

🧠 Summary Table
| Hyperparameter   | Meaning                          | Typical Value  | Notes                         |
| ---------------- | -------------------------------- | -------------- | ----------------------------- |
| `r`              | Rank of low-rank matrices        | 4–16           | Controls parameter efficiency |
| `alpha`          | Scaling factor for LoRA update   | 16–32          | Higher = more influence       |
| `target_modules` | Model layers to inject LoRA into | model-specific | Must match architecture       |

#### High level explanation of LoRA
Using Llama 3.1 with 8B weights – far too much for us to train on a **GPU**

- Llama 3.1 8B architecture consists of **32 groups of modules** stacked on top of each other, called **'Llama Decoder Layers'**
- Each has self-attention layers, multi-layer perceptron layers, SiLU activation and layer norm
- These parameters take up 32GB memory

✅ Transformer 中常見的四大組件功能
| 模組             | 作用           | 舉例                |
| -------------- | ------------ | ----------------- |
| Self-Attention | 理解詞與詞的關聯     | 「上課」→關聯「學校」、「今天」  |
| MLP            | 更深層抽象語意      | 理解句子語意是「學習活動」     |
| SiLU           | 平滑激活，維持非線性能力 | 對負值仍有微小響應         |
| LayerNorm      | 穩定訓練、提高學習效率  | 確保每層輸出均值為 0、方差為 1 |

🔍 1. Self-Attention（自注意力機制）
- 讓模型在處理一個詞時，可以「參考句子中其他所有詞」的資訊，並給每個詞一個注意力權重。
- Example：「小明今天去學校上課。」
    - 在處理「上課」這個詞時，模型會計算「學校」和「今天」對「上課」的重要性。

🔍 2. Multi-Layer Perceptron (MLP, 多層感知機)
把 attention 的輸出（例如：「學校」和「上課」的重要性）經過 MLP 處理後，轉化成更深層次語意
- 這是一個學習活動
- 發生在今天
- 主詞是小明

🔍 3. SiLU Activation（平滑激活函數）
- SiLU 是一種平滑的激活函數，能讓模型在處理負值時仍有微小響應。
- 這有助於模型在處理複雜語意時，保持非線性能力。
- Example z = -2, -1, 0, 1, 2
    - SiLU(z) = z / (1 + exp(-z))
    - 對於 z = -2，SiLU(-2) ≈ -0.238
    - 對於 z = 0，SiLU(0) = 0
    - 對於 z = 1，SiLU(1) ≈ 0.731
    - 對於 z = 2，SiLU(2) ≈ 1.761

🔍 4. Layer Normalization（層標準化）
- 標準化每一層的輸出，使其均值為 0、方差為 1，以穩定訓練過程。
- 這有助於提高模型的學習效率，讓模型更快收斂。
- Example 假設一層輸出值為 [1.5, 2.0, -3.0, 0.5]，透過 LayerNorm 會：
    - 將其平均值轉成 0
    - 將標準差轉成 1
    - 輸出變成標準化後的 [0.5, 1.0, -2.5, 0.0]

#### LoRA
- LoRA (Low-Rank Adaptation) 是一種輕量級的微調方法，專門用於大型模型。
- 它通過在模型的特定層中插入低秩矩陣來實現微調，這樣可以大幅減少需要更新的參數數量。
- LoRA 的核心思想是將模型的權重矩陣分解為兩個低秩矩陣的乘積，這樣可以在不改變原始模型結構的情況下進行微調。 
- 這樣做的好處是：
    - **節省計算資源**：只需更新少量參數，減少訓練時間和內存需求。
    - **提高效率**：在有限的資源下仍能達到良好的微調效果。

- Steps:
    1. Freeze the weights - we will not optimize them
    2. Select target modules - choose the layers where LoRA will be applied, called **target modules**
    3. Create new "adaptor" matrices with lower dimensions, fewer parameters, called **low-rank adaptors**.
    4. Inject these **low-rank adaptors** (2 LoRA matrices for each adaptor) into the **target modules**

Architecture example: Llama 3.1 8B model
```code
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention( => 1
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): LlamaMLP( => 2
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU() => 3
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False) => 4
)
```
#### Quantization – the Q in QLoRA

Even the 8B variants are enormous
- 8 Billion * 32 bits = 32GB
- Intuition: keep the number of weights but reduce their precision
- Model performance is worse, but the impact is surprisingly small
- Reduce to 8 bits, or even to 4 bits
    - Technical note 1: 4 bits are interpreted as float, not int
    - Technical note 2: the adaptor matrices are still 32 bit

#### QLoRA code examples
```code
quant_config = BitsAndBytesConfig(
    load_in_8bit=True, 
    # Add this line to enable CPU offload if GPU memory is insufficient
    llm_int8_enable_fp32_cpu_offload=True
)
```

```code
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")
```    

📊 LoRA vs QLoRA 比較表
| 項目          | LoRA (Low-Rank Adaptation)                       | QLoRA (Quantized LoRA)                                                    |
| ----------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| 🔤 全名       | Low-Rank Adaptation                              | Quantized Low-Rank Adaptation                                             |
| 🎯 目的       | 微調時只訓練少量低秩矩陣，節省參數量                               | 在 LoRA 基礎上**再加上 4-bit 量化，節省更多記憶體**                                        |
| 🧠 原始模型參數處理 | 凍結，不訓練                                           | 凍結 + **量化為 4-bit**                                                            |
| 🧮 精度       | 32-bit float                                     | 4-bit quantized float for base model (LoRA adapters still 32-bit)         |
| 💾 記憶體消耗    | 減少 **80–90%**                                        | 減少 **95–98%**（甚至能在 T4 GPU 訓練 13B 模型）                                          |
| 🧪 模型表現     | 表現接近 full fine-tune                              | 表現稍降但非常接近 LoRA（且記憶體壓縮極大）                                                  |
| 📦 可應用模型    | 任意大型 Transformer 模型                              | 通常用於 Hugging Face 支援的 4-bit 量化模型                                          |
| 🧰 工具套件支持   | 🤝 [`peft`](https://github.com/huggingface/peft) | 🤝 `peft` + [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) |
| 📈 常見應用     | 微調 GPT、LLaMA、Mistral 等                           | 在 Colab / RTX 卡上微調 LLaMA 3、Mistral 7B / 13B 等                             |

#### Three Essential Hyperparameters For LoRA Fine-Tuning

- 🕶 r
    - The rank, or how many dimensions in the low-rank matrices
    - RULE OF THUMB:
        Start with 8, then double to 16, then 32, until diminishing returns

- 🧭 Alpha
    - A scaling factor that multiplies the lower rank matrices
    - RULE OF THUMB:
        Twice the value of r (2 * r)

- 🎯 Target Modules
    - Which layers of the neural network are adapted
    - RULE OF THUMB:
        Target the attention head layers

🔝 Priority List of Target Modules for LoRA (Head Layers First)
| Priority | Module Name                           | Description                                                                              |
| -------- | ------------------------------------- | ---------------------------------------------------------------------------------------- |
| 1️⃣      | **`q_proj`** / `query`                    | Projects input into query vectors. **Crucial** for attention.                                |
| 2️⃣      | **`v_proj`** / `value`                    | Projects input into value vectors. **Also essential**.                                       |
| 3️⃣      | `k_proj` / `key`                      | Often included with `q_proj` and `v_proj`, but less critical alone.                      |
| 4️⃣      | `out_proj` / `dense`                  | Final projection after attention head. Helps adapt attention output.                     |
| 5️⃣      | `fc1`, `fc2` / `mlp`                  | Optional: feedforward MLP blocks. Helpful for task adaptation.                           |
| 6️⃣      | `gate_proj` / `up_proj` / `down_proj` | Used in more modern architectures like LLaMA2/3. Often included for complete MLP tuning. |

https://colab.research.google.com/drive/1y2Wf-4-QZeHzWLPehyFNTIV8fSJhNz1y

```code Fine-Tuning LoRA model
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 4096)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=4096, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=4096, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=4096, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=4096, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=32, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
              (up_proj): Linear4bit(in_features=4096, out_features=14336, bias=False)
              (down_proj): Linear4bit(in_features=14336, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((4096,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
    )
  )
)
```

http://localhost:8888/lab/workspaces/auto-A/tree/week7/day1-loRA_sampleL4.ipynb
http://localhost:8888/lab/workspaces/auto-A/tree/week7/day1-qlora_introL4.ipynb

#### Refer to Hugging Face Models
- https://huggingface.co/ed-donner/pricer-2024-09-13_13.04.39/tree/main
- adapter_model.safetensors (109MB)

#### Size of Weights in MB
- 32,000 – Llama 3.1 8B
- 9,000 – Quantized to 8 bit
- 5,600 – Quantized to 4 bit
- 109 – QLoRA with r=32
