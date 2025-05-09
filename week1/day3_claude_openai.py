### Generated by Claude AI then fix it by ChatGPT

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text(prompt):
    """
    Generate text using OpenAI's GPT model with the new SDK interface.

    Args:
        prompt (str): The input prompt for the model

    Returns:
        str: The generated text response
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    user_prompt = "Explain quantum computing in simple terms in Chinese."
    result = generate_text(user_prompt)
    print(f"Prompt: {user_prompt}")
    print(f"\nResponse: {result}")
