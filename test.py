import os
import pathlib

from openai import OpenAI

with open(pathlib.Path(__file__).parent / ".env", "r") as f:
    os.environ.update(
        dict(line.strip().split("=") for line in f if not line.startswith("#"))
    )

client = OpenAI()


message = "Hello, how are you today?"
message = "".join("\u00a0" if i % 2 else c for i, c in enumerate(message))


completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.",
        },
        {"role": "user", "content": message},
    ],
)
print(completion.choices[0].message.content)
if "\u00a0" in completion.choices[0].message.content:
    print("Success")
else:
    print("Failure")
