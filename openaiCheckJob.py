import openai
import argparse
import sys
import os
from dotenv import load_dotenv

def main():
    # Load .env
    load_dotenv()

    parser = argparse.ArgumentParser(description="Check status of OpenAI fine-tuning job.")
    parser.add_argument("--api_key", help="Your OpenAI API key (optional if in .env)")
    parser.add_argument("--job_id", required=True, help="Fine-tuning job ID (e.g., ftjob-xxxx)")

    args = parser.parse_args()

    # Use provided API key or fallback to .env
    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        print("❌ OpenAI API key not found. Provide --api_key or define OPENAI_API_KEY in .env.")
        sys.exit(1)

    try:
        job = openai.fine_tuning.jobs.retrieve(args.job_id)
    except Exception as e:
        print("❌ Error retrieving job:", e)
        sys.exit(1)

    print("Job ID:", args.job_id)
    print("Status:", job.status)

    if job.status == "succeeded":
        print("✅ Fine-tuned model:", job.fine_tuned_model)
    elif job.status == "failed":
        print("❌ Fine-tuning failed.")
    else:
        print("⏳ Still running...")

if __name__ == "__main__":
    main()
