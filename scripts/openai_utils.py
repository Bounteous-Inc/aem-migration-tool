import os
import openai

def load_openai_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY environment variable not set.')
    openai.api_key = api_key

def get_embedding(text, model="text-embedding-3-small"):
    load_openai_key()
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def get_gpt_completion(prompt, model="gpt-4o", max_tokens=256):
    load_openai_key()
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()