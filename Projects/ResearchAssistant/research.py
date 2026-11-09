import concurrent.futures
import json
import os
from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
SEARCH_SERVICE_URL = "http://127.0.0.1:8000/search"
MODEL = "qwen3:8b"


def scrape_url(url: str) -> str:
    """Safely fetches and strips HTML tags from a single URL."""
    try:
        response = requests.get(
            url,
            timeout=5,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "nav"]
        ):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)[:2500]
    except Exception:
        return ""


def main():
    query = input("\nQuestion / URL / PDF:\n").strip()
    if not query:
        return

    web_context = ""

    # ========================
    # PDF MODE
    # ========================
    if query.lower().startswith("pdf:"):
        pdf_path = query[4:].strip()
        print("\nReading PDF...\n")
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    web_context += text + "\n"
        except Exception as e:
            print("PDF Error:", e)
            return

    # ========================
    # URL MODE
    # ========================
    elif query.lower().startswith("url:"):
        url = query[4:].strip()
        print("\nReading URL...\n")
        web_context = scrape_url(url)

    # ========================
    # SEARCH MODE
    # ========================
    else:
        # Skip search for basic conversational greetings
        if len(query.split()) <= 2 and query.lower() in [
            "hi",
            "hello",
            "hey",
            "hello there",
        ]:
            web_context = "User is greeting. Reply politely."
        else:
            print("\nSearching web via local Search Service...\n")
            try:
                # Call local multi-tier search service
                response = requests.get(
                    SEARCH_SERVICE_URL,
                    params={"q": query, "max_results": 5},
                    timeout=10,
                )
                response.raise_for_status()
                search_data = response.json()

                provider_used = search_data.get("provider", "Unknown Provider")
                search_results = search_data.get("results", [])

                print(f"Results retrieved using provider: [{provider_used}]")
                
                urls = [r.get("url") for r in search_results if r.get("url")]

                print("Fetching web pages concurrently...")
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=5
                ) as executor:
                    scraped_texts = list(executor.map(scrape_url, urls))

                for idx, (result, text) in enumerate(
                    zip(search_results, scraped_texts), start=1
                ):
                    title = result.get("title", "")
                    url = result.get("url", "")
                    snippet = result.get("snippet", "")
                    content = text if text else snippet

                    web_context += f"\nSOURCE [{idx}]: {title}\nURL: {url}\nCONTENT:\n{content}\n"

            except Exception as e:
                print("Search Service Error:", e)
                return

    # ========================
    # QWEN PROMPT
    # ========================
    prompt = f"""You are a technical research assistant.

Instructions:
- Use ONLY the supplied sources.
- Mention disagreements between sources if any.
- Explain technical concepts clearly.

Source Material:
{web_context}

User Request:
{query}

Provide:
1. Direct Answer
2. Summary
3. Key Points
4. Sources Used
"""

    print("\nGenerating answer...\n")

    try:
        # Stream response directly to terminal for lower latency
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "options": {"num_ctx": 8192},  # Pass expanded context limit
                "stream": True,
            },
            stream=True,
        )

        print("\n=== ANSWER ===\n")
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                print(chunk.get("response", ""), end="", flush=True)
        print("\n")

    except Exception as e:
        print("Ollama API Error:", e)


if __name__ == "__main__":
    main()