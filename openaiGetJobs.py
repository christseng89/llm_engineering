import openai
import os
from dotenv import load_dotenv
from datetime import datetime

# Load the API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

# Fetch fine-tuning jobs
jobs = openai.fine_tuning.jobs.list(limit=50).data

model_ids = []
print("Available Fine-Tuned Models:\n")

for idx, job in enumerate(jobs):
    model_id = job.fine_tuned_model  # Only present if succeeded
    job_id = job.id
    base_model = job.model
    created_at = datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')
    status = job.status

    if model_id:
        model_ids.append(model_id)
        print(f"{idx}: Model: {model_id}")
        print(f"   Job ID: {job_id}")
        print(f"   Base Model: {base_model}")
        print(f"   Status: {status}")
        print(f"   Created At: {created_at}\n")

if model_ids:
    while True:
        try:
            choice = int(input(f"Enter the index of the model you want to use (0–{len(model_ids) - 1}): "))
            if 0 <= choice < len(model_ids):
                selected_model = model_ids[choice]
                print(f"\n✅ You selected model:\n{selected_model}")
                break
            else:
                print("❌ Index out of range. Try again.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
else:
    print("⚠️ No completed fine-tuning models found.")
