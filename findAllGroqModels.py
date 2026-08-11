import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
models_url = "https://api.groq.com/openai/v1/models"
chat_url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(models_url, headers=headers)
data = response.json()

if "data" in data:
    models = sorted(data["data"], key=lambda x: x["id"])
    print("Testing models usage:\n")
    for model in models:
        model_id = model["id"]
        # Skip whisper models in chat completion test as they are audio-only
        if "whisper" in model_id.lower():
            print(f"- {model_id:<40} [Skipped Chat Completion Test - Audio Model]")
            continue
            
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        res = requests.post(chat_url, headers=headers, json=payload)
        if res.status_code == 200:
            print(f"- {model_id:<40} [Success] (Status 200)")
        else:
            print(f"- {model_id:<40} [Failed] (Status {res.status_code}): {res.text}")
else:
    print(data)


