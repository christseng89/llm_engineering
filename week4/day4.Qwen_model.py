import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"  # ✅ No CUDA on Surface Pro 11

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/CodeQwen1.5-7B-Chat",
    torch_dtype=torch.float32,         # ✅ Use float32 for CPU
    device_map=None                    # ✅ Disable auto device mapping
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

prompt = "Write a quicksort algorithm in python."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)  # ✅ Move input to CPU

print("Torch in progress on CPU...")
with torch.inference_mode():  # ✅ Better performance and avoids autograd overhead
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=64
    )

# Remove the prompt portion
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
