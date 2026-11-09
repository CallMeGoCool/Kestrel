import json
import time
import requests
from datetime import datetime

BASE = "http://127.0.0.1:11434"
MODEL = "qwen3:8b"

def check_loaded_model():
    response = requests.get(
        f"{BASE}/api/ps",
        timeout=10,
    )
    response.raise_for_status()

    print("\nCURRENTLY LOADED MODELS")
    print(json.dumps(response.json(), indent=2))

def run_test(context_size):
    print("\n" + "=" * 70)
    print(f"CONTROLLED GENERATION TEST — num_ctx={context_size}")
    print("=" * 70)

    payload = {
        "model": MODEL,
        "prompt": (
            "Return one JSON object with exactly these keys: "
            '"status", "shot_count", "generation_required". '
            'Use status="ok", shot_count=3, generation_required=true.'
        ),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
            "num_ctx": context_size,
        },
    }

    started = time.perf_counter()

    try:
        response = requests.post(
            f"{BASE}/api/generate",
            json=payload,
            timeout=300,
        )

        elapsed = time.perf_counter() - started

        print(f"\nHTTP status: {response.status_code}")
        print(f"Wall time: {elapsed:.2f} seconds")

        response.raise_for_status()
        body = response.json()

        print("\nMODEL RESPONSE")
        print(body.get("response", "<missing response>"))

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
            print(f"{key}: {body.get(key)}")

        print("\nCURRENTLY LOADED MODELS AFTER REQUEST")
        check_loaded_model()

    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"\nTEST FAILED after {elapsed:.2f}s")
        print(f"{type(exc).__name__}: {exc}")

        check_loaded_model()

run_test(8192)
