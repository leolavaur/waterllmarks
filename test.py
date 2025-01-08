import os
import pathlib

from openai import OpenAI

with open(pathlib.Path(__file__).parent / ".env", "r") as f:
    os.environ.update(
        dict(line.strip().split("=") for line in f if not line.startswith("#"))
    )

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)
print(completion.choices[0].message)
