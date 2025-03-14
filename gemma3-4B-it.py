#pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cpu",
    torch_dtype=torch.bfloat16
)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are Advanced Technical assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/content/worked5.png"},
            {"type": "text", "text": "whats right role"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0][0]["generated_text"][-1]["content"])
