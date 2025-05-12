from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/codegemma-7b-it"
dtype = torch.float32  # ✅ Use float32 for CPU

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
)

# Sample chat input
chat = [
    { "role": "user", "content": "Write a hello world program" },
]

# Create chat prompt text
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

print(prompt)
