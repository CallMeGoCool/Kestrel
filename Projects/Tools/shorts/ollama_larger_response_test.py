import json
import time
import requests

url = "http://127.0.0.1:11434/api/generate"

payload = {
    "model": "qwen3:8b",
    "prompt": (
        "Return a JSON object with a key named shots containing exactly "
        "10 shot objects. Each object must have shot_id, visual_description, "
        "positive_prompt, negative_prompt, composition_mode, and asset_notes. "
        "Write detailed, distinct values for every field. "
        "Do not omit any shot."
    ),
    "stream": False,
    "format": "json",
    "think": False,
    "options": {
        "temperature": 0,
        "num_predict": 1200,
        "num_ctx": 8192,
    },
}

print("Starting controlled larger-response test...")
start = time.perf_counter()

try:
    response = requests.post(url, json=payload, timeout=300)
    elapsed = time.perf_counter() - start

    print(f"\nHTTP status: {response.status_code}")
    print(f"Wall time: {elapsed:.2f} seconds")

    response.raise_for_status()
    result = response.json()

    print("\nOLLAMA TIMING")
    for key in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "done",
        "done_reason",
    ):
        print(f"{key}: {result.get(key)}")

    text = result.get("response", "")
    print(f"\nResponse characters: {len(text)}")

    try:
        parsed = json.loads(text)
        print(f"JSON valid: yes")
        print(f"Shots returned: {len(parsed.get('shots', []))}")
    except json.JSONDecodeError as exc:
        print(f"JSON valid: no — {exc}")

except requests.exceptions.Timeout:
    print(f"\nTimed out after {time.perf_counter() - start:.2f} seconds")
except Exception as exc:
    print(f"\nError: {type(exc).__name__}: {exc}")
