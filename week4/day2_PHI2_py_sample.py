from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")

print("✅ Model loaded once and kept warm.")

# Loop to handle multiple prompts
while True:
    prompt = input("Enter your prompt (or type 'exit'): ")
    if prompt.lower() == 'exit':
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🔁 Response:\n", response)
