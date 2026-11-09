import os
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ==========================================
#  API KEYS & CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")

app = FastAPI(
    title="Multi-Tier Web Search Service",
    description="Unified API interface with sequential fallbacks supporting Open WebUI and standalone scripts."
)

# ==========================================
# UNIFIED RESPONSE MODELS
# ==========================================
class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str

class SearchResponse(BaseModel):
    provider: str
    query: str
    count: int
    results: List[SearchResultItem]

# Open WebUI Specific Models
class OpenWebUISearchRequest(BaseModel):
    query: str
    count: Optional[int] = 5

class OpenWebUISearchItem(BaseModel):
    link: str
    title: str
    snippet: str

# ==========================================
# PROVIDER IMPLEMENTATIONS
# ==========================================

def search_tavily(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 1: Tavily Search API"""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_answer": False
    }
    resp = requests.post(url, json=payload, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    for item in data.get("results", []):
        results.append(SearchResultItem(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", "")
        ))
    return results


def search_exa(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 2: Exa AI Search API"""
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "numResults": max_results,
        "useAutoprompt": True
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        results.append(SearchResultItem(
            title=item.get("title") or item.get("url", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet") or f"Exa score: {item.get('score', '')}"
        ))
    return results


def search_langsearch(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 3: LangSearch Web Search API"""
    url = "https://api.langsearch.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "count": max_results,
        "freshness": "noLimit",
        "summary": True
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    results = []
    items = data.get("data", {}).get("webPages", {}).get("value", []) or data.get("results", [])
    for item in items:
        results.append(SearchResultItem(
            title=item.get("name") or item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet") or item.get("summary", "")
        ))
    return results


def search_firecrawl(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 4: Firecrawl Search API"""
    url = "https://api.firecrawl.dev/v2/search"
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "limit": max_results
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("data", {}).get("web", []):
        results.append(SearchResultItem(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description") or item.get("markdown", "")[:200]
        ))
    return results


def search_serpapi(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 5: SerpApi Search API"""
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": max_results
    }
    resp = requests.get(url, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("organic_results", [])[:max_results]:
        results.append(SearchResultItem(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", "")
        ))
    return results


def search_github(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 6: GitHub Repository Search API"""
    url = "https://api.github.com/search/repositories"
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github+json"
    }
    params = {
        "q": query,
        "per_page": max_results
    }
    resp = requests.get(url, headers=headers, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("items", []):
        description = item.get("description") or "No description provided."
        stars = item.get("stargazers_count", 0)
        lang = item.get("language") or "Unknown"
        snippet = f"[{lang}] ⭐ {stars} | {description}"
        
        results.append(SearchResultItem(
            title=f"GitHub: {item.get('full_name', '')}",
            url=item.get("html_url", ""),
            snippet=snippet
        ))
    return results


def search_arxiv(query: str, max_results: int) -> List[SearchResultItem]:
    """Priority 7: ArXiv Open Academic Search API"""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    resp = requests.get(url, params=params, timeout=8)
    resp.raise_for_status()

    # Parse XML response
    root = ET.fromstring(resp.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        link = entry.find("atom:id", ns)

        clean_title = title.text.replace("\n", " ").strip() if title is not None else "ArXiv Paper"
        clean_summary = summary.text.replace("\n", " ").strip()[:250] + "..." if summary is not None else ""
        paper_url = link.text.strip() if link is not None else ""

        results.append(SearchResultItem(
            title=f"ArXiv: {clean_title}",
            url=paper_url,
            snippet=clean_summary
        ))
    return results


def search_local_fallback(query: str, max_results: int) -> List[SearchResultItem]:
    """Emergency Fallback: Local Google Search Scraper"""
    from googlesearch import search
    
    results = []
    for url in search(query, num_results=max_results):
        results.append(SearchResultItem(
            title="Local Fallback Result",
            url=url,
            snippet="Retrieved via local fallback scraper."
        ))
    return results

# ==========================================
# MULTI-TIER EXECUTION PIPELINE
# ==========================================

PROVIDERS = [
    ("Tavily", search_tavily),
    ("Exa", search_exa),
    ("LangSearch", search_langsearch),
    ("Firecrawl", search_firecrawl),
    ("SerpApi", search_serpapi),
    ("GitHub", search_github),
    ("ArXiv", search_arxiv),
    ("Local Google Scraper", search_local_fallback)
]

@app.get("/search", response_model=SearchResponse)
def execute_search_get(
    q: str = Query(..., description="Search query string"),
    max_results: int = Query(5, ge=1, le=20, description="Max results to return")
):
    """GET Endpoint: Used by scripts like research.py or simple curl calls."""
    errors = []

    for name, search_fn in PROVIDERS:
        try:
            results = search_fn(q, max_results)
            if results:
                return SearchResponse(
                    provider=name,
                    query=q,
                    count=len(results),
                    results=results
                )
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            continue

    raise HTTPException(
        status_code=503,
        detail={"error": "All search providers failed", "logs": errors}
    )


@app.post("/search", response_model=List[OpenWebUISearchItem])
def execute_search_post(payload: OpenWebUISearchRequest):
    """POST Endpoint: Required by Open WebUI's External Web Search engine."""
    errors = []
    count = payload.count or 5

    for name, search_fn in PROVIDERS:
        try:
            results = search_fn(payload.query, count)
            if results:
                return [
                    OpenWebUISearchItem(
                        link=item.url,
                        title=item.title,
                        snippet=item.snippet
                    )
                    for item in results
                ]
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            continue

    return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_service:app", host="0.0.0.0", port=8000, reload=True)