# Ensure you have the python-dotenv and openai libraries installed:
# pip install python-dotenv openai

import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text(prompt):
    """
    Generates text using the OpenAI API based on the given prompt.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or another suitable engine
            prompt=prompt,
            max_tokens=150,  # Adjust as needed
            n=1,             # Generate one completion
            stop=None,       # Stop generating when this token is encountered
            temperature=0.7, # Controls randomness (0.0 - 1.0)
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        print(f"An OpenAI error occurred: {e}")
        return None

if __name__ == "__main__":
    user_prompt = "Write a short poem about a rainy day."
    generated_poem = generate_text(user_prompt)

    if generated_poem:
        print("Generated Poem:")
        print(generated_poem)

    user_prompt_2 = "Translate 'Hello, how are you?' to French."
    translated_text = generate_text(user_prompt_2)

    if translated_text:
        print("\nTranslation:")
        print(translated_text)