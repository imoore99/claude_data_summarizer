import os
from anthropic import Anthropic

print("API key found:", bool(os.environ.get("ANTHROPIC_API_KEY")))  # Should print "True"
print("First 5 chars:", os.environ.get("ANTHROPIC_API_KEY", "NOT FOUND")[:5])  # Should show first 5 chars of your key


client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

resp = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=200,
    messages=[
        {"role": "user", "content": "In one short paragraph, explain what this API does."}
    ],
)

text = "".join(block.text for block in resp.content if block.type == "text")
print(text)
